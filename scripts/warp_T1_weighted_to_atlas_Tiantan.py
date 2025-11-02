#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量将患者的 Aligned/T1_registered_to_target_weighted.nii.gz warp 到指定 Atlas 空间

参考 src/warp_to_atlas.py 的实现风格，并借鉴 scripts/register_mra_images_Tiantan.py 的结构：
- 统一参数解析与日志
- 批量患者处理，支持并行
- 健壮的路径与文件校验
- 在形状不一致时，先重采样到 Atlas 网格

输出：每位患者在 output_dir/<patient_id>/AtlasWarped 下生成：
- warped_image_to_atlas.nii.gz
- warped_segmentation_to_atlas.nii.gz
- moving_to_atlas_displacement.nii.gz
- warp_manifest.json

使用示例：
  python scripts/warp_T1_weighted_to_atlas_Tiantan.py \
    -s /data/MARVAL/Precise/Baseline \
    -o /data/MARVAL/Precise/Baseline \
    --atlas_image /path/to/atlas.nii.gz \
    --model_path ./models/model_cvpr.pt \
    -j 4 --overwrite

依赖：SimpleITK、PyTorch、项目内 src/warp_to_atlas.py
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import SimpleITK as sitk
except ImportError as e:
    print(f"错误：缺少 SimpleITK 依赖 {e}")
    print("请安装：pip install SimpleITK")
    sys.exit(1)

# 确保可以导入 src/warp_to_atlas
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from src.warp_to_atlas import warp_image_and_segmentation_to_atlas
except Exception as e:
    print(f"错误：无法导入 src.warp_to_atlas：{e}")
    print("请确认项目结构完整且存在 src/warp_to_atlas.py")
    sys.exit(1)


def setup_logging(output_dir: Optional[str] = None, log_level: str = "INFO") -> logging.Logger:
    """
    设置日志记录配置
    """
    logger = logging.getLogger("WarpToAtlas")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_dir:
        log_dir = Path(output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"warp_to_atlas_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_patient_list(source_dir: str) -> List[str]:
    """
    获取患者列表：遍历子目录，过滤仅包含 Aligned/T1_registered_to_target_weighted.nii.gz 的患者
    """
    src = Path(source_dir)
    patients: List[str] = []
    for d in src.iterdir():
        if not d.is_dir():
            continue
        aligned = d / "Aligned" / "T1_registered_to_target_weighted.nii.gz"
        if aligned.exists():
            patients.append(d.name)
    return sorted(patients)


def _read_image(path: Path) -> sitk.Image:
    return sitk.ReadImage(str(path))


def _resample_to_reference(moving: sitk.Image, reference: sitk.Image, interp: str) -> sitk.Image:
    """
    将 moving 重采样到 reference 的网格（尺寸/spacing/方向/原点），interp 为 'linear' 或 'nearest'
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.Transform())  # identity
    if interp == 'linear':
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interp == 'nearest':
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        raise ValueError(f"不支持的插值类型: {interp}")
    return resampler.Execute(moving)


def ensure_same_shape_as_atlas(t1_path: Path, seg_path: Path, atlas_path: Path, temp_dir: Path) -> Tuple[Path, Path, bool]:
    """
    确保 T1 和 SEG 与 atlas 形状一致；若不一致则重采样到 atlas 网格。
    返回：(new_t1_path, new_seg_path, resampled_flag)
    """
    temp_dir.mkdir(parents=True, exist_ok=True)

    t1_img = _read_image(t1_path)
    seg_img = _read_image(seg_path) if seg_path.exists() else None
    atlas_img = _read_image(atlas_path)

    same_size = t1_img.GetSize() == atlas_img.GetSize()
    if same_size and (seg_img is None or seg_img.GetSize() == atlas_img.GetSize()):
        return t1_path, seg_path, False

    # 重采样 T1 到 atlas
    t1_rs = _resample_to_reference(t1_img, atlas_img, 'linear')
    t1_rs_path = temp_dir / "T1_weighted_resampled_to_atlas_shape.nii.gz"
    sitk.WriteImage(t1_rs, str(t1_rs_path))

    # 重采样/或创建 SEG 到 atlas
    if seg_img is not None:
        seg_rs = _resample_to_reference(seg_img, atlas_img, 'nearest')
    else:
        # 创建与 atlas 同尺寸的零分割
        seg_rs = sitk.Image(atlas_img.GetSize(), sitk.sitkUInt16)
        seg_rs.SetSpacing(atlas_img.GetSpacing())
        seg_rs.SetOrigin(atlas_img.GetOrigin())
        seg_rs.SetDirection(atlas_img.GetDirection())
    seg_rs_path = temp_dir / "SEG_resampled_to_atlas_shape.nii.gz"
    sitk.WriteImage(seg_rs, str(seg_rs_path))

    return t1_rs_path, seg_rs_path, True


def write_manifest(path: Path, content: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def process_one_patient(
    source_dir: str,
    patient_id: str,
    atlas_image: str,
    model_path: str,
    output_dir: str,
    divisor: int,
    device: Optional[str],
    overwrite: bool = False,
    log_level: str = "INFO",
) -> Tuple[str, bool, Optional[str]]:
    """
    处理单个患者：warp 加权 T1 和 SEG 到 atlas
    返回：(patient_id, success, error_message)
    """
    # 建立患者级输出与日志
    patient_out_dir = Path(output_dir) / patient_id / "AtlasWarped"
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"WarpToAtlas_{patient_id}")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        fh = logging.FileHandler(patient_out_dir / f"warp_{patient_id}.log", encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 输入路径
    aligned_dir = Path(source_dir) / patient_id / "Aligned"
    t1_weighted_path = aligned_dir / "T1_registered_to_target_weighted.nii.gz"
    seg_path = aligned_dir / "SEG_registered_to_target.nii.gz"

    # 目标 atlas
    atlas_path = Path(atlas_image)

    # 跳过逻辑
    out_img = patient_out_dir / 'warped_image_to_atlas.nii.gz'
    out_seg = patient_out_dir / 'warped_segmentation_to_atlas.nii.gz'
    out_def = patient_out_dir / 'moving_to_atlas_displacement.nii.gz'
    manifest_path = patient_out_dir / 'warp_manifest.json'

    if (not overwrite) and out_img.exists() and out_seg.exists() and out_def.exists() and manifest_path.exists():
        logger.info("输出已存在且未开启覆盖，跳过处理")
        return patient_id, True, None

    # 校验存在性
    if not t1_weighted_path.exists():
        msg = f"缺少加权 T1 文件: {t1_weighted_path}"
        logger.error(msg)
        return patient_id, False, msg
    if not atlas_path.exists():
        msg = f"Atlas 文件不存在: {atlas_path}"
        logger.error(msg)
        return patient_id, False, msg

    # 形状检查与必要时重采样
    temp_dir = patient_out_dir / "TempForAtlas"
    moving_img_path, moving_seg_path, did_resample = ensure_same_shape_as_atlas(
        t1_path=t1_weighted_path,
        seg_path=seg_path,
        atlas_path=atlas_path,
        temp_dir=temp_dir,
    )

    logger.info(f"开始 warp 到 atlas：患者 {patient_id}")
    logger.info(f"模型: {model_path}")
    logger.info(f"设备: {device or 'auto'}，padding divisor: {divisor}")
    try:
        paths = warp_image_and_segmentation_to_atlas(
            model_path=model_path,
            moving_image_path=str(moving_img_path),
            moving_segmentation_path=str(moving_seg_path),
            atlas_image_path=str(atlas_path),
            output_dir=str(patient_out_dir),
            divisor=divisor,
            device=device,
        )

        manifest = {
            "patient_id": patient_id,
            "source_dir": str(Path(source_dir).resolve()),
            "aligned_t1_weighted": str(t1_weighted_path.resolve()),
            "aligned_seg": str(seg_path.resolve()) if seg_path.exists() else None,
            "atlas_image": str(atlas_path.resolve()),
            "model_path": str(Path(model_path).resolve()),
            "did_resample_to_atlas_shape": did_resample,
            "outputs": paths,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "divisor": divisor,
            "device": device or ("cuda" if _has_cuda() else "cpu"),
        }
        write_manifest(manifest_path, manifest)
        logger.info(f"完成 warp：图像 {paths['warped_image']}，分割 {paths['warped_segmentation']}")
        return patient_id, True, None
    except Exception as e:
        msg = f"warp 失败：{e}"
        logger.error(msg)
        return patient_id, False, msg


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="批量将加权 T1 (Aligned/T1_registered_to_target_weighted.nii.gz) warp 到指定 Atlas 空间",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
示例：
  python scripts/warp_T1_weighted_to_atlas_Tiantan.py \
    -s /data/MARVAL/Precise/Baseline \
    -o /data/MARVAL/Precise/Baseline \
    --atlas_image /path/to/atlas.nii.gz \
    --model_path ./models/model_cvpr.pt \
    -j 4 --overwrite
        """
    )

    parser.add_argument('-s', '--source_dir', type=str, required=True, help='源数据目录（包含各患者子目录）')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='输出目录根（默认写回到 source_dir/patient_id/AtlasWarped）')
    parser.add_argument('--atlas_image', type=str, required=True, help='Atlas 图像路径 (nii.gz)')
    parser.add_argument('--model_path', type=str, default=str(Path(REPO_ROOT) / 'models' / 'model_cvpr.pt'), help='预训练 MultiMorph 权重路径')
    parser.add_argument('--divisor', type=int, default=16, help='空间维度 padding 的最小公倍数 (默认:16)')
    parser.add_argument('--device', type=str, default=None, help='Torch 设备字符串 (例如 cuda:0 或 cpu；默认自动)')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='并行进程数 (默认: 1)')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出')
    parser.add_argument('--max_patients', type=int, default=None, help='仅处理前 N 个患者（测试用）')
    parser.add_argument('--patients_file', type=str, default=None, help='患者ID列表文件，每行一个')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'], help='日志级别')

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"错误：源目录不存在：{source_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else source_dir
    atlas_image = Path(args.atlas_image)
    if not atlas_image.exists():
        print(f"错误：Atlas 文件不存在：{atlas_image}")
        return 1

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误：模型权重不存在：{model_path}")
        return 1

    logger = setup_logging(str(output_dir), args.log_level)
    logger.info("=== Warp T1 Weighted to Atlas (Tiantan) ===")
    logger.info(f"源目录: {source_dir}")
    logger.info(f"输出根: {output_dir}")
    logger.info(f"Atlas: {atlas_image}")
    logger.info(f"模型: {model_path}")
    logger.info(f"jobs: {args.jobs}, overwrite: {args.overwrite}, log_level: {args.log_level}")

    # 构建患者列表
    if args.patients_file:
        with open(args.patients_file, 'r', encoding='utf-8') as f:
            patients = [line.strip() for line in f if line.strip()]
    else:
        patients = get_patient_list(str(source_dir))

    if args.max_patients is not None:
        patients = patients[:args.max_patients]

    if not patients:
        logger.error("未找到需处理的患者（缺少 Aligned/T1_registered_to_target_weighted.nii.gz）")
        return 1

    logger.info(f"待处理患者数：{len(patients)}")

    processed_ok = 0
    processed_fail = 0

    if args.jobs > 1:
        logger.info("并行处理启动...")
        work_items = []
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            for pid in patients:
                fut = executor.submit(
                    process_one_patient,
                    str(source_dir),
                    pid,
                    str(atlas_image),
                    str(model_path),
                    str(output_dir),
                    int(args.divisor),
                    args.device,
                    bool(args.overwrite),
                    str(args.log_level),
                )
                work_items.append(fut)

            for fut in as_completed(work_items):
                pid, ok, err = fut.result()
                if ok:
                    processed_ok += 1
                    logger.info(f"[OK] {pid}")
                else:
                    processed_fail += 1
                    logger.error(f"[FAIL] {pid}: {err}")
    else:
        for pid in patients:
            pid, ok, err = process_one_patient(
                str(source_dir),
                pid,
                str(atlas_image),
                str(model_path),
                str(output_dir),
                int(args.divisor),
                args.device,
                bool(args.overwrite),
                str(args.log_level),
            )
            if ok:
                processed_ok += 1
                logger.info(f"[OK] {pid}")
            else:
                processed_fail += 1
                logger.error(f"[FAIL] {pid}: {err}")

    logger.info("=== 处理完成 ===")
    logger.info(f"成功: {processed_ok}, 失败: {processed_fail}")
    return 0 if processed_fail == 0 else 1


if __name__ == '__main__':
    exit(main())