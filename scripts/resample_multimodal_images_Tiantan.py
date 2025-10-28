#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
医学影像批量重采样脚本

该脚本遍历患者文件夹，将 T1 图像、原始 MRA 图像和 MRA 血管分割
批量重采样到指定的目标体素网格 (如 256x256x128)。

它使用“参考网格”方法来确保重采样后的所有图像完美对齐。
"""

import ants
import os
import time
import multiprocessing
import argparse
import logging
import logging.handlers
import sys
from queue import Empty  # 用于队列监听器
from pathlib import Path
from typing import Tuple

def worker_log_config(log_queue):
    """配置工作进程的日志记录器"""
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(logging.handlers.QueueHandler(log_queue))
    logger.setLevel(logging.INFO)


def process_patient(patient_dir: str, overwrite: bool, target_shape: Tuple[int, int, int]):
    """
    处理单个患者的重采样任务。
    
    返回: (patient_id, status_str, error_message)
    """
    logger = logging.getLogger()
    patient_id = os.path.basename(patient_dir)
    log_prefix = f"[{patient_id}]"

    try:
        # 1. 定义输入文件路径
        t1_in_path = Path(patient_dir) / "T1" / "T1_warped_to_MRA.nii.gz"
        seg_in_path = Path(patient_dir) / "Predictions" / "MRA_vessel_pred.nii.gz"
        mra_in_path = Path(patient_dir) / "TOF-MRA" / "MRA.nii.gz" # <-- 新增输入
        
        # 2. 定义输出文件路径
        output_dir = Path(patient_dir) / "Resampled"
        t1_out_path = output_dir / "T1_warped_resampled.nii.gz"
        seg_out_path = output_dir / "MRA_vessel_pred_resampled.nii.gz"
        mra_out_path = output_dir / "MRA_resampled.nii.gz" # <-- 新增输出
        
        # 3. 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 4. 断点继续 (Overwrite 检查)
        # 只有当所有文件都存在时才跳过
        if (t1_out_path.exists() and 
            seg_out_path.exists() and 
            mra_out_path.exists() and 
            not overwrite):
            logger.info(f"{log_prefix} SKIPPED: 所有三个输出文件已存在。")
            return (patient_id, 'skipped', None)

        # 5. 检查输入文件是否存在
        required_files = {
            'T1': t1_in_path, 
            'Seg': seg_in_path, 
            'MRA': mra_in_path # <-- 新增检查
        }
        missing_files = []
        for name, path in required_files.items():
            if not path.exists():
                missing_files.append(f"{name} ({path.name})")
        
        if missing_files:
            error_msg = f"缺少文件: {', '.join(missing_files)}"
            logger.warning(f"{log_prefix} FAILED: {error_msg}")
            return (patient_id, 'failed', error_msg)

        # 6. --- 核心重采样逻辑 ---
        logger.info(f"{log_prefix} LOADING 图像...")
        t1_img = ants.image_read(str(t1_in_path))
        seg_img = ants.image_read(str(seg_in_path))
        mra_img = ants.image_read(str(mra_in_path)) # <-- 新增加载

        # 7. 创建目标参考网格 (Reference Grid)
        # 我们使用 seg_img 作为基础来定义物理空间 (任选一个输入图像都行)
        logger.info(f"{log_prefix} CREATING 参考网格 (Shape: {target_shape})...")
        reference_grid = ants.resample_image(
            seg_img,
            target_shape,
            use_voxels=True,
            interpolator='nearestNeighbor'
        )
        
        # 8. 重采样 T1 到参考网格
        logger.info(f"{log_prefix} RESAMPLING T1 (Interpolator: bSpline)...")
        t1_resampled = ants.resample_image_to_target(
            t1_img,
            reference_grid,
            interpolator='bSpline'
        )

        # 9. 重采样 Segmentation 到参考网格
        logger.info(f"{log_prefix} RESAMPLING Seg (Interpolator: nearestNeighbor)...")
        seg_resampled = ants.resample_image_to_target(
            seg_img,
            reference_grid,
            interpolator='nearestNeighbor'
        )
        
        # 10. 重采样 MRA 到参考网格 (新增)
        logger.info(f"{log_prefix} RESAMPLING MRA (Interpolator: bSpline)...")
        mra_resampled = ants.resample_image_to_target(
            mra_img,
            reference_grid,
            interpolator='bSpline' # <-- 对强度图像使用 bSpline
        )

        # 11. 保存结果
        logger.info(f"{log_prefix} SAVING 结果...")
        ants.image_write(t1_resampled, str(t1_out_path))
        ants.image_write(seg_resampled, str(seg_out_path))
        ants.image_write(mra_resampled, str(mra_out_path)) # <-- 新增保存
        
        logger.info(f"{log_prefix} SUCCESS.")
        return (patient_id, 'success', None)

    except Exception as e:
        error_msg = f"发生意外错误: {e}"
        logger.exception(f"{log_prefix} FAILED: {error_msg}") # 包含堆栈跟踪
        return (patient_id, 'failed', str(e))


def batch_resample_images(base_dir: str, overwrite: bool, num_workers: int, target_shape: Tuple[int, int, int]):
    """
    使用多进程并行处理所有患者。
    """
    logger = logging.getLogger()
    
    try:
        patient_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    except FileNotFoundError:
        logger.error(f"错误：基础目录 '{base_dir}' 不存在。")
        return

    if not patient_folders:
        logger.warning(f"错误：在 '{base_dir}' 中未找到任何患者子文件夹。")
        return

    logger.info(f"总共找到 {len(patient_folders)} 个患者文件夹。")
    logger.info(f"目标 Shape: {target_shape}")
    logger.info(f"启动 {num_workers} 个并行工作进程...")
    logger.info(f"Overwrite 模式: {'ON' if overwrite else 'OFF'}")

    # 创建任务列表
    tasks = [(patient_dir, overwrite, target_shape) for patient_dir in patient_folders]

    total_start_time = time.time()
    
    # 使用 multiprocessing.Pool
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=worker_log_config,
        initargs=(log_queue,)
    ) as pool:
        results = pool.starmap(process_patient, tasks)

    total_end_time = time.time()

    # --- 汇总结果 ---
    logger.info("\n" + "="*30 + " 批处理摘要 " + "="*30)
    logger.info(f"总耗时: {(total_end_time - total_start_time) / 60:.2f} 分钟")

    success_count = 0
    skipped_count = 0
    failed_list = []

    for patient_id, status, error_msg in results:
        if status == 'success':
            success_count += 1
        elif status == 'skipped':
            skipped_count += 1
        elif status == 'failed':
            failed_list.append((patient_id, error_msg))

    logger.info(f"成功处理: {success_count}")
    logger.info(f"跳过 (已存在): {skipped_count}")
    logger.info(f"失败: {len(failed_list)}")
    
    if failed_list:
        logger.warning("\n--- 失败详情 ---")
        for patient_id, error_msg in failed_list:
            logger.warning(f"  - {patient_id}: {error_msg}")
    logger.info("="*74)


# --- 主程序入口 ---
if __name__ == "__main__":
    
    # 1. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="批量将 T1、MRA 和 MRA分割 重采样到目标 Shape")
    
    parser.add_argument("-d", "--base_dir", 
                        type=str, 
                        required=True, 
                        help="包含所有患者子文件夹的基础目录 (必需)")
    
    parser.add_argument("-n", "--num_workers", 
                        type=int, 
                        default=max(1, os.cpu_count() // 2), 
                        help="用于处理的并行进程数 (默认: 系统核心数的一半)")
    
    parser.add_argument("-O", "--overwrite", 
                        action="store_true", 
                        help="如果设置此项, 将覆盖已存在的输出文件 (默认: False)")

    parser.add_argument("-l", "--log_file", 
                        type=str, 
                        default=None, 
                        help="指定日志文件路径 (默认: 在 base_dir 下创建 'batch_resample_...log')")
    
    parser.add_argument("--shape", 
                        type=str, 
                        default="256,256,128", 
                        help='目标 shape (e.g., "256,256,128") (默认: 256,256,128)')

    args = parser.parse_args()

    # 2. 设置多进程启动方法
    multiprocessing.set_start_method('fork', force=True)

    # 3. 解析 --shape 参数
    try:
        shape_list = [int(s.strip()) for s in args.shape.split(',')]
        if len(shape_list) != 3:
            raise ValueError
        target_shape_tuple = tuple(shape_list)
    except ValueError:
        print(f"错误: --shape 参数格式不正确: '{args.shape}'。必须是三个逗号分隔的整数, e.g., '256,256,128'")
        sys.exit(1)

    # 4. *** 设置日志系统 ***
    log_queue = multiprocessing.Queue()
    log_file_path = args.log_file
    if log_file_path is None:
        log_file_name = f'batch_resample_{target_shape_tuple[0]}x{target_shape_tuple[1]}x{target_shape_tuple[2]}.log'
        if os.path.isdir(args.base_dir):
            log_file_path = os.path.join(args.base_dir, log_file_name)
        else:
            log_file_path = log_file_name
            print(f"警告：base_dir 无效，日志将写入: {log_file_path}")

    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    console_handler = logging.StreamHandler(sys.stdout)
    log_format = '%(asctime)s - %(processName)-12s - [%(levelname)-8s] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    listener = logging.handlers.QueueListener(log_queue, file_handler, console_handler)
    listener.start()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))
    
    # 5. 验证 base_dir 并运行主函数
    main_logger = logging.getLogger()
    main_logger.info(f"脚本启动。日志文件位于: {log_file_path}")
    main_logger.info(f"运行参数: {args}")

    if not os.path.isdir(args.base_dir):
        main_logger.critical("="*50)
        main_logger.critical(f"错误：指定的 --base_dir 路径不存在或不是一个目录！")
        main_logger.critical(f"路径: {args.base_dir}")
        main_logger.critical("="*50)
    else:
        try:
            # 6. 使用解析到的参数调用主函数
            batch_resample_images(
                args.base_dir, 
                args.overwrite, 
                args.num_workers, 
                target_shape_tuple
            )
        except Exception as e:
            main_logger.exception("主程序发生未捕获的严重错误:")
        finally:
            # 7. 停止日志监听器
            main_logger.info("批处理完成。停止日志监听器...")
            listener.stop()