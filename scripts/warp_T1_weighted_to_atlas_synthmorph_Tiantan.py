#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 SynthMorph 将每位患者的 Aligned/T1_registered_to_target_weighted.nii.gz 注册到指定 Atlas，
并将同一变换应用到 Aligned/SEG_registered_to_target.nii.gz（若存在）。

输出到 patient/AtlasWarped 下：
- warped_image_to_atlas.nii.gz
- warped_segmentation_to_atlas.nii.gz（若有分割）
- moving_to_atlas_transform.nii.gz 或 moving_to_atlas_transform.lta（取决于模式）
- warp_manifest.json

参考：scripts/register_t1_to_mra_synthmorph_Tiantan.py 与 SynthMorph CLI 用法

示例：
  python scripts/warp_T1_weighted_to_atlas_synthmorph_Tiantan.py 
    -s /data/MARVAL/Precise/Baseline 
    --atlas_image /path/to/atlas.nii.gz 
    --synthmorph_cmd ./synthmorph 
    -j 4 --overwrite --reg_strength 0.25 --mode joint

依赖：SynthMorph CLI（例如 mri_synthmorph 或本仓库 ./synthmorph）
"""

import os
import sys
import json
import time
import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed


def setup_logging(output_dir: Optional[str] = None, log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("WarpToAtlas_SynthMorph")
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
        log_file = log_dir / f"warp_to_atlas_synthmorph_{timestamp}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_patient_list(source_dir: str) -> List[str]:
    src = Path(source_dir)
    patients: List[str] = []
    for d in src.iterdir():
        if not d.is_dir():
            continue
        aligned = d / "Aligned" / "T1_registered_to_target_weighted.nii.gz"
        if aligned.exists():
            patients.append(d.name)
    return sorted(patients)


def _is_executable_available(cmd: str) -> bool:
    p = Path(cmd)
    if p.exists():
        return os.access(str(p), os.X_OK)
    return shutil.which(cmd) is not None


def _run(cmd: List[str], logger: logging.Logger) -> Tuple[bool, Optional[str], Optional[str]]:
    logger.debug(f"运行命令: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if res.stdout:
            logger.debug(res.stdout)
        if res.stderr:
            logger.debug(res.stderr)
        return True, res.stdout, res.stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"命令失败: {e}")
        if e.stdout:
            logger.debug(e.stdout)
        if e.stderr:
            logger.error(e.stderr)
        return False, e.stdout, e.stderr
    except FileNotFoundError:
        logger.error("找不到 SynthMorph 可执行文件，请检查 --synthmorph_cmd 或 PATH")
        return False, None, None


def process_one_patient(
    source_dir: str,
    patient_id: str,
    atlas_image: str,
    output_dir: str,
    synthmorph_cmd: str,
    mode: str,
    reg_strength: float,
    overwrite: bool,
    log_level: str,
) -> Tuple[str, bool, Optional[str]]:
    """
    使用 SynthMorph 注册 T1 到 atlas，并应用同一变换到 SEG。
    返回：(patient_id, success, error_message)
    """
    patient_out_dir = Path(output_dir) / patient_id / "AtlasWarped"
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"WarpToAtlas_SynthMorph_{patient_id}")
    logger.setLevel(getattr(logging, log_level.upper()))
    if not logger.handlers:
        fh = logging.FileHandler(patient_out_dir / f"warp_{patient_id}_synthmorph.log", encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    aligned_dir = Path(source_dir) / patient_id / "Aligned"
    t1_weighted_path = aligned_dir / "T1_registered_to_target_weighted.nii.gz"
    seg_path = aligned_dir / "SEG_registered_to_target.nii.gz"
    atlas_path = Path(atlas_image)

    out_img = patient_out_dir / 'warped_image_to_atlas.nii.gz'
    # transform 后缀根据模式选择
    transform_suffix = '.lta' if mode == 'affine' else '.nii.gz'
    out_transform = patient_out_dir / f'moving_to_atlas_transform{transform_suffix}'
    out_seg = patient_out_dir / 'warped_segmentation_to_atlas.nii.gz'
    manifest_path = patient_out_dir / 'warp_manifest.json'

    if (not overwrite) and out_img.exists() and out_transform.exists() and manifest_path.exists() and (not seg_path.exists() or out_seg.exists()):
        logger.info("输出已存在且未开启覆盖，跳过处理")
        return patient_id, True, None

    if not t1_weighted_path.exists():
        msg = f"缺少加权 T1 文件: {t1_weighted_path}"
        logger.error(msg)
        return patient_id, False, msg
    if not atlas_path.exists():
        msg = f"Atlas 文件不存在: {atlas_path}"
        logger.error(msg)
        return patient_id, False, msg
    if not _is_executable_available(synthmorph_cmd):
        msg = f"SynthMorph 命令不可用: {synthmorph_cmd}"
        logger.error(msg)
        return patient_id, False, msg

    # 构建 register 命令
    register_cmd: List[str] = [synthmorph_cmd, 'register']
    if mode == 'affine':
        register_cmd += ['-m', 'affine']
    else:
        # joint 默认包含可变形部分，设置正则强度
        if reg_strength is not None:
            register_cmd += ['-r', str(reg_strength)]
    register_cmd += ['-t', str(out_transform), '-o', str(out_img), str(t1_weighted_path), str(atlas_path)]

    logger.info(f"开始注册：患者 {patient_id}，模式 {mode}，正则 {reg_strength}")
    ok, _, err = _run(register_cmd, logger)
    if not ok:
        return patient_id, False, f"注册失败：{err}"

    # 对分割应用同一变换（若存在）
    if seg_path.exists():
        apply_cmd: List[str] = [synthmorph_cmd, 'apply']
        # 分割采用最近邻插值
        apply_cmd += ['-m', 'nearest', str(out_transform), str(seg_path), str(out_seg)]
        logger.info("应用变换到分割标签（nearest）")
        ok2, _, err2 = _run(apply_cmd, logger)
        if not ok2:
            return patient_id, False, f"应用到分割失败：{err2}"
    else:
        logger.warning("未找到分割文件，跳过分割变换应用")

    # 写清单
    manifest = {
        "patient_id": patient_id,
        "source_dir": str(Path(source_dir).resolve()),
        "aligned_t1_weighted": str(t1_weighted_path.resolve()),
        "aligned_seg": str(seg_path.resolve()) if seg_path.exists() else None,
        "atlas_image": str(atlas_path.resolve()),
        "synthmorph_cmd": synthmorph_cmd,
        "mode": mode,
        "reg_strength": reg_strength,
        "outputs": {
            "warped_image": str(out_img.resolve()),
            "warped_segmentation": str(out_seg.resolve()) if out_seg.exists() else None,
            "transform": str(out_transform.resolve()),
        },
        "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("完成注册与应用")
    return patient_id, True, None


def main():
    parser = argparse.ArgumentParser(
        description="使用 SynthMorph 将加权 T1 Warp 到指定 Atlas 并应用到分割",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
示例：
  python scripts/warp_T1_weighted_to_atlas_synthmorph_Tiantan.py \
    -s /data/MARVAL/Precise/Baseline \
    --atlas_image /path/to/atlas.nii.gz \
    --synthmorph_cmd ./synthmorph \
    -j 4 --overwrite --reg_strength 0.25 --mode joint
        """
    )

    parser.add_argument('-s', '--source_dir', type=str, required=True, help='源数据目录（包含各患者子目录）')
    parser.add_argument('--atlas_image', type=str, required=True, help='Atlas 图像路径 (nii.gz)')
    parser.add_argument('--synthmorph_cmd', type=str, default='./synthmorph', help='SynthMorph 命令（例如 ./synthmorph 或 mri_synthmorph）')
    parser.add_argument('--mode', type=str, default='joint', choices=['joint', 'affine'], help='注册模式：joint(默认) 或 affine')
    parser.add_argument('--reg_strength', type=float, default=0.25, help='变形正则强度（joint 模式有效，默认 0.25）')
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

    atlas_image = Path(args.atlas_image)
    if not atlas_image.exists():
        print(f"错误：Atlas 文件不存在：{atlas_image}")
        return 1

    if not _is_executable_available(args.synthmorph_cmd):
        print(f"错误：SynthMorph 命令不可用：{args.synthmorph_cmd}")
        print("请确认：1) 在 scripts/ 目录使用 ./synthmorph；或 2) 安装 mri_synthmorph 并将其在 PATH 中")
        return 1

    logger = setup_logging(str(source_dir), args.log_level)
    logger.info("=== Warp T1 Weighted to Atlas (SynthMorph Tiantan) ===")
    logger.info(f"源目录: {source_dir}")
    logger.info(f"Atlas: {atlas_image}")
    logger.info(f"SynthMorph: {args.synthmorph_cmd}, 模式: {args.mode}, 正则: {args.reg_strength}")
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
                    str(source_dir),
                    str(args.synthmorph_cmd),
                    str(args.mode),
                    float(args.reg_strength),
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
                str(source_dir),
                str(args.synthmorph_cmd),
                str(args.mode),
                float(args.reg_strength),
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