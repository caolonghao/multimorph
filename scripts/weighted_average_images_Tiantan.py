#!/usr/bin/env python3
"""
Weighted Average Images for Tiantan Dataset

This script performs weighted averaging of aligned T1 images and their corresponding 
segmentations, generating weighted T1 files and a CSV file compatible with multimorph.

Based on the data flow described in README.md and using resample_baseline_to_target_shape.py 
as reference.

Author: AI Assistant
Date: 2025-10-29
"""

import csv
import logging
import argparse
from pathlib import Path
import SimpleITK as sitk
import numpy as np


def setup_logging(log_level=logging.INFO):
    """设置日志配置"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def normalize_image_to_01(image):
    """
    将图像归一化到[0,1]范围
    
    Args:
        image: SimpleITK图像对象
        
    Returns:
        归一化后的SimpleITK图像对象
    """
    # 获取图像数组
    image_array = sitk.GetArrayFromImage(image)
    
    # 计算最小值和最大值
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    
    # 避免除零错误
    if max_val == min_val:
        normalized_array = np.zeros_like(image_array)
    else:
        # 归一化到[0,1]
        normalized_array = (image_array - min_val) / (max_val - min_val)
    
    # 创建新的SimpleITK图像
    normalized_image = sitk.GetImageFromArray(normalized_array.astype(np.float32))
    normalized_image.CopyInformation(image)  # 复制原图像的元信息
    
    return normalized_image


def weighted_sum_images(image, segmentation, alpha=0.5):
    """
    对图像和分割图像进行加权求和
    
    Args:
        image: SimpleITK图像对象（已归一化）
        segmentation: SimpleITK分割图像对象
        alpha: 加权系数，image的权重 (0-1之间)
        
    Returns:
        加权求和后的SimpleITK图像对象
    """
    # 获取图像数组
    image_array = sitk.GetArrayFromImage(image)
    seg_array = sitk.GetArrayFromImage(segmentation)
    
    # 确保分割图像是二值化的（0或1）
    seg_array = (seg_array > 0).astype(np.float32)
    
    # 执行加权求和: result = alpha * image + (1-alpha) * seg
    weighted_sum_array = alpha * image_array + (1 - alpha) * seg_array
    
    # 创建新的SimpleITK图像
    weighted_sum_image = sitk.GetImageFromArray(weighted_sum_array)
    weighted_sum_image.CopyInformation(image)  # 复制原图像的元信息
    
    return weighted_sum_image


def safe_load_image(file_path: Path):
    """
    安全加载图像文件
    
    Args:
        file_path: 图像文件路径 (Path)
        
    Returns:
        SimpleITK图像对象，如果加载失败返回None
    """
    try:
        if not file_path.exists():
            logging.warning(f"文件不存在: {file_path}")
            return None
        image = sitk.ReadImage(str(file_path))
        logging.debug(f"成功加载图像: {file_path}")
        return image
    except Exception as e:
        logging.error(f"加载图像失败 {file_path}: {str(e)}")
        return None


def process_patient_weighted_average(patient_dir: Path, alpha: float = 0.5):
    """
    处理单个患者的加权平均
    
    Args:
        patient_dir: 患者数据目录路径 (Path)
        alpha: 加权系数
        
    Returns:
        tuple: (success, input_t1_path, input_seg_path, output_weighted_path)
    """
    patient_id = patient_dir.name
    
    # 统一在 patient_id / "Aligned" 下进行读写
    aligned_dir = patient_dir / "Aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    # 定义输入文件路径（位于 Aligned 下）
    input_t1_path = aligned_dir / "T1_registered_to_target.nii.gz"
    input_seg_path = aligned_dir / "SEG_registered_to_target.nii.gz"
    
    # 定义输出文件路径（同样位于 Aligned 下）
    output_weighted_path = aligned_dir / "T1_registered_to_target_weighted.nii.gz"
    
    # 检查输出文件是否已存在
    if output_weighted_path.exists():
        logging.info(f"跳过已存在的文件: {output_weighted_path}")
        return True, input_t1_path, input_seg_path, output_weighted_path
    
    # 加载T1图像
    t1_image = safe_load_image(input_t1_path)
    if t1_image is None:
        logging.error(f"无法加载T1图像: {input_t1_path}")
        return False, input_t1_path, input_seg_path, output_weighted_path
    
    # 加载分割图像
    seg_image = safe_load_image(input_seg_path)
    if seg_image is None:
        logging.error(f"无法加载分割图像: {input_seg_path}")
        return False, input_t1_path, input_seg_path, output_weighted_path
    
    try:
        # 归一化T1图像到[0,1]
        normalized_t1 = normalize_image_to_01(t1_image)
        
        # 执行加权求和
        weighted_image = weighted_sum_images(normalized_t1, seg_image, alpha)
        
        # 保存加权平均结果
        sitk.WriteImage(weighted_image, str(output_weighted_path))
        logging.info(f"成功生成加权平均图像: {output_weighted_path}")
        
        return True, input_t1_path, input_seg_path, output_weighted_path
        
    except Exception as e:
        logging.error(f"处理患者 {patient_id} 时出错: {str(e)}")
        return False, input_t1_path, input_seg_path, output_weighted_path


def weighted_average_images_tiantan(base_dir: Path, alpha: float = 0.5):
    """
    对Tiantan数据集的对齐T1图像和分割进行加权平均
    
    Args:
        base_dir: 基础目录路径（包含患者子目录）(Path)
        alpha: 加权系数，T1图像的权重 (0-1之间)
        
    Returns:
        tuple: (output_dir, metadata_file, processed_count, failed_count)
    """
    # 基础目录存在性在 main 中已校验
    
    # 存储处理结果用于生成CSV
    csv_records = []
    processed_count = 0
    failed_count = 0
    
    # 获取所有患者目录
    patient_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith('P')]
    
    # 按患者ID排序
    patient_dirs.sort(key=lambda p: p.name)
    
    logging.info(f"找到 {len(patient_dirs)} 个患者目录")
    
    # 处理每个患者
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        logging.info(f"处理患者: {patient_id}")
        
        success, input_t1_path, input_seg_path, output_weighted_path = process_patient_weighted_average(
            patient_dir, alpha
        )
        
        if success:
            # 记录成功处理的文件路径
            csv_records.append({
                'img_path': str(input_t1_path.resolve()),
                'segmentation_path': str(input_seg_path.resolve()),
                'weighted_T1_img_path': str(output_weighted_path.resolve()),
            })
            processed_count += 1
        else:
            failed_count += 1
    
    # 生成CSV文件
    metadata_file = base_dir / "metadata.csv"
    with open(metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
        if csv_records:
            fieldnames = ['img_path', 'segmentation_path', 'weighted_T1_img_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据行
            for record in csv_records:
                writer.writerow(record)
    
    logging.info(f"处理完成: 成功 {processed_count} 个，失败 {failed_count} 个")
    logging.info(f"CSV文件已生成: {metadata_file}")
    
    return base_dir, metadata_file, processed_count, failed_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="对Tiantan数据集的对齐T1图像和分割进行加权平均"
    )
    
    parser.add_argument(
        "--base_dir", 
        type=str, 
        default="/data/MARVAL/Precise/Baseline",
        help="基础目录路径（包含患者子目录），输出也写回各患者的Aligned下"
    )
    
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.5,
        help="加权系数，T1图像的权重 (0-1之间，默认: 0.5)"
    )
    
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)
    
    # 验证参数
    if not (0 <= args.alpha <= 1):
        logging.error("alpha参数必须在0-1之间")
        return 1
    
    base_dir = Path(args.base_dir)

    if not base_dir.exists():
        logging.error(f"基础目录不存在: {base_dir}")
        return 1
    
    # 打印配置信息
    logging.info("=== Weighted Average Images for Tiantan Dataset ===")
    logging.info(f"基础目录: {base_dir}")
    logging.info(f"加权系数 (alpha): {args.alpha}")
    logging.info(f"日志级别: {args.log_level}")
    
    try:
        # 执行加权平均处理
        base_dir, metadata_file, processed_count, failed_count = weighted_average_images_tiantan(
            base_dir, args.alpha
        )
        
        # 打印结果摘要
        logging.info("=== 处理结果摘要 ===")
        logging.info(f"基础目录: {base_dir}")
        logging.info(f"CSV文件: {metadata_file}")
        logging.info(f"成功处理: {processed_count} 个患者")
        logging.info(f"处理失败: {failed_count} 个患者")
        
        if failed_count > 0:
            logging.warning(f"有 {failed_count} 个患者处理失败，请检查日志")
            return 1
        
        logging.info("所有患者处理完成！")
        return 0
        
    except Exception as e:
        logging.error(f"程序执行出错: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())