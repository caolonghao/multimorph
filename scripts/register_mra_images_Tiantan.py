#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学影像配准脚本 - MRA/T1 图像配准工具

该脚本用于将多个患者的MRA影像和T1影像配准到指定的目标患者的MRA空间。
它首先计算 MRA 到 目标MRA 的变换，然后将此变换应用到 MRA 和 T1 影像上。
支持刚性、仿射和SyNRA配准，使用ANTsPy库。

作者: 自动生成 (基于用户模板修改)
日期: 2025-10-28
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import traceback
import shutil
import json

try:
    import ants
    import numpy as np
except ImportError as e:
    print(f"错误：缺少必要的依赖库 {e}")
    print("请安装以下依赖：")
    print("pip install antspyx numpy")
    sys.exit(1)

# 设置日志配置
def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """
    设置日志记录配置
    
    Args:
        output_dir: 输出目录路径
        log_level: 日志级别
    
    Returns:
        配置好的logger对象
    """
    # 创建日志目录
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成日志文件名（包含时间戳）
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"registration_{timestamp}.log"
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建logger
    logger = logging.getLogger('MRA_Registration')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_patient_list(source_dir: str) -> List[str]:
    """
    获取需要配准的患者列表
    
    Args:
        source_dir: 源数据目录
    
    Returns:
        患者ID列表
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"源目录不存在: {source_dir}")
    
    # 获取所有患者目录 (假设所有子目录都是患者)
    patient_dirs = [d.name for d in source_path.iterdir() if d.is_dir()]
    
    # 排序
    patient_list = sorted(patient_dirs)
    
    return patient_list

def validate_patient_data(source_dir: str, patient_id: str) -> Tuple[bool, str]:
    """
    验证患者数据文件是否存在
    
    Args:
        source_dir: 源数据目录
        patient_id: 患者ID
    
    Returns:
        (是否有效, 错误信息)
    """
    patient_dir = Path(source_dir) / patient_id
    
    if not patient_dir.exists():
        return False, f"患者目录不存在: {patient_dir}"
    
    # 检查必需的文件 (根据新文件结构)
    mra_file = patient_dir / "Resampled" / "MRA_resampled.nii.gz"
    t1_file = patient_dir / "Resampled" / "T1_warped_resampled.nii.gz"
    seg_file = patient_dir / "Resampled" / "MRA_vessel_pred_resampled.nii.gz"
    
    if not mra_file.exists():
        return False, f"MRA文件不存在: {mra_file}"
    
    if not t1_file.exists():
        return False, f"T1文件不存在: {t1_file}"
    
    if not seg_file.exists():
        return False, f"分割文件不存在: {seg_file}"
    
    return True, ""

def load_image_safe(image_path: str, logger: logging.Logger) -> Optional[ants.ANTsImage]:
    """
    安全地加载影像文件
    
    Args:
        image_path: 影像文件路径
        logger: 日志记录器
    
    Returns:
        ANTs影像对象或None
    """
    try:
        image = ants.image_read(str(image_path))
        logger.debug(f"成功加载影像: {image_path}")
        logger.debug(f"影像信息 - 尺寸: {image.shape}, 间距: {image.spacing}")
        return image
    except Exception as e:
        logger.error(f"加载影像失败 {image_path}: {e}")
        return None

def perform_registration(
    moving_image: ants.ANTsImage,
    fixed_image: ants.ANTsImage,
    registration_type: str,
    logger: logging.Logger
) -> Optional[Dict]:
    """
    执行配准操作 (MRA-to-MRA)
    
    Args:
        moving_image: 待配准影像 (MRA)
        fixed_image: 目标影像 (Target MRA)
        registration_type: 配准类型 ('rigid', 'affine', 'SyNRA')
        logger: 日志记录器
    
    Returns:
        配准结果字典或None
    """
    try:
        logger.info(f"开始执行MRA-to-MRA的 {registration_type} 配准...")
        logger.info(f"移动影像尺寸: {moving_image.shape}, 间距: {moving_image.spacing}")
        logger.info(f"固定影像尺寸: {fixed_image.shape}, 间距: {fixed_image.spacing}")
        start_time = time.time()
        
        reg_result = ants.registration(
            fixed=fixed_image,
            moving=moving_image,
            type_of_transform=registration_type,
            verbose=True  # 显示配准进度
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"配准完成，耗时: {elapsed_time:.2f}秒")
        
        # 检查配准结果是否有效
        if 'fwdtransforms' not in reg_result or not reg_result['fwdtransforms']:
             logger.error("配准失败：未生成 'fwdtransforms'。")
             return None
             
        return reg_result
        
    except Exception as e:
        logger.error(f"配准失败: {e}")
        logger.error(traceback.format_exc())
        return None

def apply_transform_to_image(
    moving_image: ants.ANTsImage,
    transform_list: List,
    fixed_image: ants.ANTsImage,
    interpolation: str = 'bSpline',
    logger: logging.Logger = None
) -> Optional[ants.ANTsImage]:
    """
    将变换应用到影像上
    
    Args:
        moving_image: 待变换影像
        transform_list: 变换列表
        fixed_image: 参考影像 (定义输出空间)
        interpolation: 插值方法
        logger: 日志记录器
    
    Returns:
        变换后的影像或None
    """
    try:
        transformed_image = ants.apply_transforms(
            fixed=fixed_image,
            moving=moving_image,
            transformlist=transform_list,
            interpolator=interpolation
        )
        
        if logger:
            logger.debug(f"成功应用变换，插值方法: {interpolation}")
            
        return transformed_image
        
    except Exception as e:
        if logger:
            logger.error(f"应用变换失败: {e}")
        return None

def save_registration_transforms(
    reg_result: Dict,
    transform_output_dir: Path,
    logger: logging.Logger
) -> Tuple[List[Path], List[Path]]:
    """
    复制并持久化 ANTs 配准产生的变换文件到目标目录，并返回保存后的文件列表。

    Args:
        reg_result: ANTs registration 返回的结果字典
        transform_output_dir: 变换文件保存目录
        logger: 日志记录器

    Returns:
        (fwd_files, inv_files): 前向与反向变换文件的保存路径列表
    """
    transform_output_dir.mkdir(parents=True, exist_ok=True)

    fwd_files: List[Path] = []
    inv_files: List[Path] = []

    # 保存前向变换
    for i, tf in enumerate(reg_result.get('fwdtransforms', [])):
        src = Path(str(tf))
        if not src.exists():
            logger.warning(f"前向变换文件不存在: {src}")
            continue
        dst = transform_output_dir / f"fwd_{i}_{src.name}"
        try:
            shutil.copy(src, dst)
            fwd_files.append(dst)
            logger.debug(f"保存前向变换: {dst}")
        except Exception as e:
            logger.error(f"复制前向变换失败 {src} -> {dst}: {e}")

    # 保存反向变换
    for i, tf in enumerate(reg_result.get('invtransforms', [])):
        src = Path(str(tf))
        if not src.exists():
            logger.warning(f"反向变换文件不存在: {src}")
            continue
        dst = transform_output_dir / f"inv_{i}_{src.name}"
        try:
            shutil.copy(src, dst)
            inv_files.append(dst)
            logger.debug(f"保存反向变换: {dst}")
        except Exception as e:
            logger.error(f"复制反向变换失败 {src} -> {dst}: {e}")

    return fwd_files, inv_files

def write_transform_manifest(
    manifest_path: Path,
    moving_patient: str,
    fixed_patient: str,
    registration_type: str,
    fwd_files: List[Path],
    inv_files: List[Path],
    image_interpolation: str,
    seg_interpolation: str,
    logger: logging.Logger,
    identity: bool = False
) -> None:
    """
    写出一个 JSON manifest，记录本次变换的关键信息，方便后续调用。

    Args:
        manifest_path: JSON 保存路径
        moving_patient: 移动影像患者ID
        fixed_patient: 固定影像患者ID
        registration_type: 配准类型
        fwd_files: 前向变换保存文件列表
        inv_files: 反向变换保存文件列表
        image_interpolation: 连续图像插值方法（MRA/T1）
        seg_interpolation: 标签图插值方法（SEG）
        logger: 日志记录器
        identity: 是否为身份变换（目标患者）
    """
    data = {
        "moving_patient": moving_patient,
        "fixed_patient": fixed_patient,
        "registration_type": registration_type,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "identity": identity,
        "image_interpolation": image_interpolation,
        "seg_interpolation": seg_interpolation,
        "fwdtransforms": [p.name for p in (fwd_files or [])],
        "invtransforms": [p.name for p in (inv_files or [])]
    }

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"变换清单已保存: {manifest_path}")
    except Exception as e:
        logger.error(f"写出变换清单失败: {e}")

def register_patient(
    source_dir: str,
    target_patient: str,
    patient_id: str,
    output_dir: str,
    registration_type: str,
    interpolation: str,
    logger: logging.Logger,
    skip_existing: bool = True
) -> bool:
    """
    配准单个患者的影像 (MRA 和 T1)
    
    Args:
        ... (参数同上) ...
    
    Returns:
        是否成功
    """
    logger.info(f"开始处理患者: {patient_id}")
    
    # 设置文件路径
    source_path = Path(source_dir)
    output_path = Path(output_dir) / patient_id / "Aligned"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # (新) 定义输入文件路径
    t1_input_path = source_path / patient_id / "Resampled" / "T1_warped_resampled.nii.gz"
    mra_input_path = source_path / patient_id / "Resampled" / "MRA_resampled.nii.gz"
    seg_input_path = source_path / patient_id / "Resampled" / "MRA_vessel_pred_resampled.nii.gz"
    
    # (新) 定义输出文件路径
    output_t1_path = output_path / "T1_registered_to_target.nii.gz"
    output_mra_path = output_path / "MRA_registered_to_target.nii.gz"
    output_seg_path = output_path / "SEG_registered_to_target.nii.gz"
    transform_dir = output_path / "Transforms"
    manifest_path = transform_dir / "transform_manifest.json"
    
    # 检查是否跳过 (检查 T1、MRA、SEG 以及变换manifest 是否都已存在)
    if skip_existing and output_t1_path.exists() and output_mra_path.exists() and output_seg_path.exists() and manifest_path.exists():
        logger.info(f"跳过已存在的患者 (T1 和 MRA): {patient_id}")
        return True
    
    # 验证数据 (确保 MRA 和 T1 都存在)
    is_valid, error_msg = validate_patient_data(source_dir, patient_id)
    if not is_valid:
        logger.error(f"患者 {patient_id} 数据验证失败: {error_msg}")
        return False
        
    try:
        if patient_id == target_patient:
            # 目标患者，直接复制 MRA 和 T1 到输出目录
            logger.info(f"患者 {patient_id} 是目标患者。")
            
            # 复制 T1
            if not output_t1_path.exists():
                shutil.copy(t1_input_path, output_t1_path)
                logger.info(f"复制目标患者 T1: {output_t1_path}")
            else:
                logger.info(f"目标患者 T1 已存在: {output_t1_path}")
                
            # 复制 MRA
            if not output_mra_path.exists():
                shutil.copy(mra_input_path, output_mra_path)
                logger.info(f"复制目标患者 MRA: {output_mra_path}")
            else:
                logger.info(f"目标患者 MRA 已存在: {output_mra_path}")

            # 复制 SEG
            if seg_input_path.exists():
                if not output_seg_path.exists():
                    shutil.copy(seg_input_path, output_seg_path)
                    logger.info(f"复制目标患者 SEG: {output_seg_path}")
                else:
                    logger.info(f"目标患者 SEG 已存在: {output_seg_path}")
            else:
                logger.warning(f"目标患者未找到 SEG 文件: {seg_input_path}")
            
            # 写入目标患者的身份变换 manifest（不含具体变换文件）
            write_transform_manifest(
                manifest_path=manifest_path,
                moving_patient=patient_id,
                fixed_patient=target_patient,
                registration_type=registration_type,
                fwd_files=[],
                inv_files=[],
                image_interpolation=interpolation,
                seg_interpolation='nearestNeighbor',
                logger=logger,
                identity=True
            )

            return True
        
        # --- 对于其他患者，需要进行配准 ---
        
        # 1. 加载目标影像（固定影像）
        target_mra_path = source_path / target_patient / "Resampled" / "MRA_resampled.nii.gz"
        fixed_image = load_image_safe(target_mra_path, logger)
        if fixed_image is None:
            logger.error(f"无法加载目标患者影像: {target_mra_path}")
            return False
        
        # 2. 加载当前患者影像（移动影像）
        moving_mra_image = load_image_safe(mra_input_path, logger)
        moving_t1_image = load_image_safe(t1_input_path, logger)
        moving_seg_image = load_image_safe(seg_input_path, logger)
        
        if moving_mra_image is None or moving_t1_image is None:
            logger.error(f"无法加载患者 {patient_id} 的 MRA 或 T1 影像文件")
            return False
        if moving_seg_image is None:
            logger.warning(f"未找到或无法加载患者 {patient_id} 的 SEG 影像，将跳过 SEG 变换: {seg_input_path}")
        
        # 3. 执行 MRA-to-MRA 配准
        reg_result = perform_registration(
            moving_mra_image, fixed_image, registration_type, logger
        )
        
        if reg_result is None:
            logger.error(f"患者 {patient_id} MRA 配准失败")
            return False
        
        # 保存变换文件与清单，便于后续调用
        logger.info("保存配准产生的变换文件与清单...")
        fwd_files, inv_files = save_registration_transforms(
            reg_result=reg_result,
            transform_output_dir=transform_dir,
            logger=logger
        )
        write_transform_manifest(
            manifest_path=manifest_path,
            moving_patient=patient_id,
            fixed_patient=target_patient,
            registration_type=registration_type,
            fwd_files=fwd_files,
            inv_files=inv_files,
            image_interpolation=interpolation,
            seg_interpolation='nearestNeighbor',
            logger=logger,
            identity=False
        )
        
        # 4. 应用变换到 T1 影像
        logger.info(f"应用变换到 T1 影像 (插值方法: {interpolation})...")
        transformed_t1 = apply_transform_to_image(
            moving_t1_image,
            reg_result['fwdtransforms'],
            fixed_image, # 参考空间是目标 MRA
            interpolation,
            logger
        )
        
        if transformed_t1 is None:
            logger.error(f"患者 {patient_id} T1 变换应用失败")
            return False
        
        # 保存 T1
        ants.image_write(transformed_t1, str(output_t1_path))
        logger.info(f"保存 T1 配准文件: {output_t1_path}")
        
        # 5. 应用变换到 MRA 影像
        logger.info(f"应用变换到 MRA 影像 (插值方法: {interpolation})...")
        transformed_mra = apply_transform_to_image(
            moving_mra_image,
            reg_result['fwdtransforms'],
            fixed_image, # 参考空间是目标 MRA
            interpolation,
            logger
        )
        
        if transformed_mra is None:
            logger.error(f"患者 {patient_id} MRA 变换应用失败")
            return False
        
        # 保存 MRA
        ants.image_write(transformed_mra, str(output_mra_path))
        logger.info(f"保存 MRA 配准文件: {output_mra_path}")

        # 6. 应用变换到 SEG 影像（使用最近邻插值以保持标签）
        if moving_seg_image is not None:
            logger.info("应用变换到 SEG 影像 (插值方法: nearestNeighbor)...")
            transformed_seg = apply_transform_to_image(
                moving_seg_image,
                reg_result['fwdtransforms'],
                fixed_image,
                'nearestNeighbor',
                logger
            )

            if transformed_seg is None:
                logger.error(f"患者 {patient_id} SEG 变换应用失败")
            else:
                ants.image_write(transformed_seg, str(output_seg_path))
                logger.info(f"保存 SEG 配准文件: {output_seg_path}")
        
        logger.info(f"患者 {patient_id} 处理完成")
        return True
        
    except Exception as e:
        logger.error(f"处理患者 {patient_id} 时发生错误: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="MRA/T1影像配准工具 - 将多个患者的MRA/T1配准到目标患者的MRA空间",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 必需指定 -s 和 -o
  python register_script.py -s ./data -o ./registered_data -t P0048
  
  # 使用 SyNRA 配准
  python register_script.py -s ./data -t P0117 -r SyNRA -o ./output --skip_existing
  
  # 仅处理5个患者用于测试
  python register_script.py -s ./data -o ./test_output -t P0151 -m 5  
        """
    )
    
    parser.add_argument(
        '-s', '--source_dir',
        type=str,
        required=True, # (新) 设为必需
        help='源数据目录路径 (必需)'
    )
    
    parser.add_argument(
        '-t', '--target_patient',
        type=str,
        default='P0048',
        help='目标患者ID (默认: P0048)'
    )
    
    parser.add_argument(
        '-o', '--output_dir',
        type=str,
        required=True,
        help='输出目录路径 (必需)'
    )
    
    parser.add_argument(
        '-r', '--registration_type',
        type=str,
        choices=['rigid', 'affine', 'SyNRA', 'SyN', 'SyNCC'], # (新) 增加了 SyN 和 SyNCC
        default='affine',
        help='配准类型 (默认: affine)。'
             ' SyNRA/SyN/SyNCC 包含非线性配准。'
    )
    
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='跳过已存在的文件 (会检查 MRA 和 T1 是否都已存在)'
    )
    
    parser.add_argument(
        '-l', '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )
    
    parser.add_argument(
        '-m', '--max_patients',
        type=int,
        default=None,
        help='最大处理患者数量（用于测试）'
    )
    
    parser.add_argument(
        '-i', '--interpolation',
        type=str,
        choices=['bSpline', 'linear', 'nearestNeighbor'],
        default='bSpline',
        help='MRA 和 T1 图像插值方法 (默认: bSpline)'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not Path(args.source_dir).exists():
        print(f"错误: 源目录不存在: {args.source_dir}")
        sys.exit(1)
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(args.output_dir, args.log_level)
    
    # 验证目标患者 (使用新的验证函数)
    is_valid, error_msg = validate_patient_data(args.source_dir, args.target_patient)
    if not is_valid:
        logger.error(f"错误: 目标患者数据无效: {error_msg}")
        sys.exit(1)
    
    # 记录开始信息
    logger.info("="*60)
    logger.info("MRA/T1 影像配准工具启动")
    logger.info("="*60)
    logger.info(f"源数据目录: {args.source_dir}")
    logger.info(f"目标患者 (MRA空间): {args.target_patient}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"配准类型: {args.registration_type}")
    logger.info(f"插值方法: {args.interpolation}")
    logger.info(f"跳过已存在文件: {args.skip_existing}")
    
    try:
        # 获取患者列表
        patient_list = get_patient_list(args.source_dir)
        
        if args.max_patients:
            patient_list = patient_list[:args.max_patients]
            logger.info(f"限制处理患者数量: {args.max_patients}")
        
        logger.info(f"需要处理的患者数量: {len(patient_list)}")
        logger.info(f"患者列表: {patient_list}")
        
        # 统计信息
        success_count = 0
        failed_count = 0
        failed_patients = []
        
        start_time = time.time()
        
        # 逐个处理患者
        for i, patient_id in enumerate(patient_list, 1):
            logger.info(f"\n进度: [{i}/{len(patient_list)}] 处理患者 {patient_id}")
            
            success = register_patient(
                args.source_dir,
                args.target_patient,
                patient_id,
                args.output_dir,
                args.registration_type,
                args.interpolation,
                logger,
                args.skip_existing
            )
            
            if success:
                success_count += 1
            else:
                failed_count += 1
                failed_patients.append(patient_id)
            
            # 显示进度
            progress = (i / len(patient_list)) * 100
            elapsed = time.time() - start_time
            if i > 0:
                estimated_total = elapsed / i * len(patient_list)
                remaining = estimated_total - elapsed
                logger.info(f"进度: {progress:.1f}% | "
                           f"成功: {success_count} | "
                           f"失败: {failed_count} | "
                           f"剩余时间: {remaining/60:.1f}分钟")
            else:
                 logger.info(f"进度: {progress:.1f}% | "
                           f"成功: {success_count} | "
                           f"失败: {failed_count}")
        
        # 输出最终统计
        total_time = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("配准完成!")
        logger.info("="*60)
        logger.info(f"总处理时间: {total_time/60:.2f}分钟")
        logger.info(f"总患者数: {len(patient_list)}")
        logger.info(f"成功: {success_count}")
        logger.info(f"失败: {failed_count}")
        
        if failed_patients:
            logger.warning(f"失败的患者: {failed_patients}")
        
        # 保存处理摘要
        summary_file = output_path / "registration_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"MRA/T1 影像配准处理摘要\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"目标患者 (MRA空间): {args.target_patient}\n")
            f.write(f"配准类型: {args.registration_type}\n")
            f.write(f"插值方法: {args.interpolation}\n")
            f.write(f"总患者数: {len(patient_list)}\n")
            f.write(f"成功: {success_count}\n")
            f.write(f"失败: {failed_count}\n")
            if failed_patients:
                f.write(f"失败患者: {', '.join(failed_patients)}\n")
        
        logger.info(f"处理摘要已保存: {summary_file}")
        
        # 退出码
        sys.exit(0 if failed_count == 0 else 1)
        
    except KeyboardInterrupt:
        logger.warning("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()