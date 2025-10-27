#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学影像配准脚本 - MRA图像配准工具

该脚本用于将多个患者的MRA影像配准到指定的目标患者影像空间。
支持刚性配准和仿射配准，使用ANTsPy库进行配准操作。

作者: 自动生成
日期: 2024
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
    
    # 获取所有患者目录
    patient_dirs = [d.name for d in source_path.iterdir() 
                   if d.is_dir() and d.name.startswith('P')]
    
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
    
    # 检查必需的文件
    mra_file = patient_dir / "aligned_mra.nii.gz"
    seg_file = patient_dir / "aligned_mra_brain_seg.nii.gz"
    
    if not mra_file.exists():
        return False, f"MRA文件不存在: {mra_file}"
    
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
    执行配准操作
    
    Args:
        moving_image: 待配准影像
        fixed_image: 目标影像
        registration_type: 配准类型 ('rigid', 'affine', 'both')
        logger: 日志记录器
    
    Returns:
        配准结果字典或None
    """
    try:
        logger.info(f"开始执行{registration_type}配准...")
        logger.info(f"移动影像尺寸: {moving_image.shape}, 间距: {moving_image.spacing}")
        logger.info(f"固定影像尺寸: {fixed_image.shape}, 间距: {fixed_image.spacing}")
        start_time = time.time()
        
        if registration_type == 'rigid':
            # 刚性配准
            logger.info("执行刚性配准...")
            reg_result = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform='Rigid',
                verbose=True  # 显示配准进度
            )
        elif registration_type == 'affine':
            # 仿射配准
            logger.info("执行仿射配准...")
            reg_result = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform='Affine',
                verbose=True  # 显示配准进度
            )
        elif registration_type == 'SyNRA':
            # 先刚性后仿射配准
            logger.info("执行组合配准（刚性+仿射）...")
            reg_result = ants.registration(
                fixed=fixed_image,
                moving=moving_image,
                type_of_transform='SyNRA',  # 包含刚性和仿射变换
                verbose=True  # 显示配准进度
            )
        else:
            raise ValueError(f"不支持的配准类型: {registration_type}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"配准完成，耗时: {elapsed_time:.2f}秒")
        logger.info(f"最终相似性度量: {reg_result['metricvalue'] if 'metricvalue' in reg_result else 'N/A'}")
        
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
        fixed_image: 参考影像
        interpolation: 插值方法 ('bSpline', 'linear', 'nearestNeighbor')
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
    配准单个患者的影像
    
    Args:
        source_dir: 源数据目录
        target_patient: 目标患者ID
        patient_id: 当前患者ID
        output_dir: 输出目录
        registration_type: 配准类型
        interpolation: 插值方法
        logger: 日志记录器
        skip_existing: 是否跳过已存在的文件
    
    Returns:
        是否成功
    """
    logger.info(f"开始处理患者: {patient_id}")
    
    # 设置文件路径
    source_path = Path(source_dir)
    output_path = Path(output_dir) / patient_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 输出文件路径
    output_t1 = output_path / "T1_aligned.nii.gz"
    
    # 检查是否跳过已存在的 T1 文件
    if skip_existing and output_t1.exists():
        logger.info(f"跳过已存在的 T1 文件: {patient_id}")
        return True
    
    # 加载 T1_moved.nii.gz
    t1_moved_path = source_path / patient_id / "T1_moved.nii.gz"
    if not t1_moved_path.exists():
        logger.warning(f"T1 文件不存在: {t1_moved_path}")
        return False
    
    t1_image = load_image_safe(t1_moved_path, logger)
    if t1_image is None:
        return False
    
    if patient_id == target_patient:
        # 目标患者，直接保存 T1
        ants.image_write(t1_image, str(output_t1))
        logger.info(f"保存目标患者 T1: {output_t1}")
        
        # 复制 MRA 和 segmentation
        mra_src = source_path / patient_id / "aligned_mra.nii.gz"
        seg_src = source_path / patient_id / "aligned_mra_brain_seg.nii.gz"
        mra_dst = output_path / "aligned_mra_resampled.nii.gz"
        seg_dst = output_path / "aligned_mra_seg_resampled.nii.gz"
        
        if mra_src.exists():
            shutil.copy(mra_src, mra_dst)
            logger.info(f"复制目标患者 MRA: {mra_dst}")
        else:
            logger.warning(f"MRA 文件不存在: {mra_src}")
        
        if seg_src.exists():
            shutil.copy(seg_src, seg_dst)
            logger.info(f"复制目标患者 segmentation: {seg_dst}")
        else:
            logger.warning(f"Segmentation 文件不存在: {seg_src}")
        
        return True
    
    # 对于其他患者，需要进行配准
    # 验证数据
    is_valid, error_msg = validate_patient_data(source_dir, patient_id)
    if not is_valid:
        logger.error(f"患者 {patient_id} 数据验证失败: {error_msg}")
        return False
    
    try:
        # 加载目标影像（固定影像）
        target_mra_path = source_path / target_patient / "aligned_mra.nii.gz"
        fixed_image = load_image_safe(target_mra_path, logger)
        if fixed_image is None:
            logger.error(f"无法加载目标患者影像: {target_mra_path}")
            return False
        
        # 加载当前患者影像（移动影像）
        moving_mra_path = source_path / patient_id / "aligned_mra.nii.gz"
        moving_seg_path = source_path / patient_id / "aligned_mra_brain_seg.nii.gz"
        
        moving_image = load_image_safe(moving_mra_path, logger)
        moving_seg = load_image_safe(moving_seg_path, logger)
        
        if moving_image is None or moving_seg is None:
            logger.error(f"无法加载患者 {patient_id} 的影像文件")
            return False
        
        # 执行配准
        reg_result = perform_registration(
            moving_image, fixed_image, registration_type, logger
        )
        
        if reg_result is None:
            logger.error(f"患者 {patient_id} 配准失败")
            return False
        
        # 应用变换到 T1
        logger.info(f"应用变换到 T1 影像 (插值方法: {interpolation})...")
        transformed_t1 = apply_transform_to_image(
            t1_image,
            reg_result['fwdtransforms'],
            fixed_image,
            interpolation,
            logger
        )
        
        if transformed_t1 is None:
            logger.error(f"患者 {patient_id} T1 变换应用失败")
            return False
        
        # 保存 T1
        ants.image_write(transformed_t1, str(output_t1))
        logger.info(f"保存 T1 对齐文件: {output_t1}")
        
        # 应用变换到原始影像
        logger.info(f"应用变换到MRA影像 (插值方法: {interpolation})...")
        transformed_mra = apply_transform_to_image(
            moving_image,
            reg_result['fwdtransforms'],
            fixed_image,
            interpolation,
            logger
        )
        
        # 应用变换到分割mask（始终使用最近邻插值保持标签完整性）
        logger.info("应用变换到分割mask (插值方法: nearestNeighbor)...")
        transformed_seg = apply_transform_to_image(
            moving_seg,
            reg_result['fwdtransforms'],
            fixed_image,
            'nearestNeighbor',
            logger
        )
        
        if transformed_mra is None or transformed_seg is None:
            logger.error(f"患者 {patient_id} 变换应用失败")
            return False
        
        # 保存结果
        logger.info("保存配准结果...")
        output_mra = output_path / "aligned_mra_resampled.nii.gz"
        output_seg = output_path / "aligned_mra_seg_resampled.nii.gz"
        ants.image_write(transformed_mra, str(output_mra))
        ants.image_write(transformed_seg, str(output_seg))
        
        logger.info(f"患者 {patient_id} 处理完成")
        logger.info(f"输出文件: {output_mra}")
        logger.info(f"输出文件: {output_seg}")
        
        return True
        
    except Exception as e:
        logger.error(f"处理患者 {patient_id} 时发生错误: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="MRA影像配准工具 - 将多个患者的影像配准到目标患者空间",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python register_mra_images.py -o ./registered_data
  python register_mra_images.py -s ./data -t P0117 -r SyNRA -o ./output
  python register_mra_images.py -t P0151 -o ./output --skip_existing
  python register_mra_images.py -o ./test_output -m 5  # 仅处理5个患者用于测试
        """
    )
    
    parser.add_argument(
        '-s', '--source_dir',
        type=str,
        default='multimorph/data/baseline_3d_data',
        help='源数据目录路径 (默认: multimorph/data/baseline_3d_data)'
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
        help='输出目录路径'
    )
    
    parser.add_argument(
        '-r', '--registration_type',
        type=str,
        choices=['rigid', 'affine', 'SyNRA'],
        default='affine',
        help='配准类型 (默认: affine)'
    )
    
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='跳过已存在的文件'
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
        help='MRA图像插值方法 (默认: bSpline)'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not Path(args.source_dir).exists():
        print(f"错误: 源目录不存在: {args.source_dir}")
        sys.exit(1)
    
    # 验证目标患者
    is_valid, error_msg = validate_patient_data(args.source_dir, args.target_patient)
    if not is_valid:
        print(f"错误: 目标患者数据无效: {error_msg}")
        sys.exit(1)
    
    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logging(args.output_dir, args.log_level)
    
    # 记录开始信息
    logger.info("="*60)
    logger.info("MRA影像配准工具启动")
    logger.info("="*60)
    logger.info(f"源数据目录: {args.source_dir}")
    logger.info(f"目标患者: {args.target_patient}")
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
            estimated_total = elapsed / i * len(patient_list)
            remaining = estimated_total - elapsed
            
            logger.info(f"进度: {progress:.1f}% | "
                       f"成功: {success_count} | "
                       f"失败: {failed_count} | "
                       f"剩余时间: {remaining/60:.1f}分钟")
        
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
            f.write(f"MRA影像配准处理摘要\n")
            f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"目标患者: {args.target_patient}\n")
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