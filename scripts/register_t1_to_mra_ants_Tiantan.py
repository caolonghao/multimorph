import ants
import os
import time
import multiprocessing
import argparse # 1. 导入 argparse

def process_patient(patient_dir, overwrite=False):
    """
    处理单个患者的注册任务。
    此函数被设计为可以被多进程并行调用。
    
    返回: (patient_id, status_str, error_message)
    """
    patient_id = os.path.basename(patient_dir)
    pid = os.getpid()  # 获取当前进程的ID，用于日志
    log_prefix = f"[PID: {pid}] [ {patient_id} ]"

    try:
        # 1. 定义文件路径
        t1_path = os.path.join(patient_dir, 'T1', 'T1.nii.gz')
        mra_path = os.path.join(patient_dir, 'Preprocessed', 'MRA', 'MRA_normalized_sum.nii.gz')
        mask_path = os.path.join(patient_dir, 'Preprocessed', 'MRA', 'MRA_brain_mask.nii.gz')
        
        output_dir = os.path.join(patient_dir, 'T1')
        output_path = os.path.join(output_dir, 'T1_warped_to_MRA.nii.gz')

        # 2. 断点继续 (Overwrite 检查)
        if os.path.exists(output_path) and not overwrite:
            print(f"{log_prefix} SKIPPED: 输出文件已存在。")
            return (patient_id, 'skipped', None)

        # 3. 检查输入文件是否存在
        required_files = {'T1': t1_path, 'MRA': mra_path, 'Mask': mask_path}
        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing_files.append(name)
        
        if missing_files:
            error_msg = f"缺少文件: {', '.join(missing_files)}"
            print(f"{log_prefix} FAILED: {error_msg}")
            return (patient_id, 'failed', error_msg)

        # 4. 加载图像
        print(f"{log_prefix} LOADING 图像...")
        t1_img = ants.image_read(t1_path)
        mra_img = ants.image_read(mra_path)
        mra_mask = ants.image_read(mask_path)

        # 5. 执行配准 (计算变换)
        print(f"{log_prefix} COMPUTING 仿射变换...")
        tx = ants.registration(
            fixed=mra_img,
            moving=t1_img,
            type_of_transform='Affine',
            mask=mra_mask
        )

        # 6. 应用变换 (Resample)
        print(f"{log_prefix} APPLYING 变换 (B-spline)...")
        t1_warped = ants.apply_transforms(
            fixed=mra_img,
            moving=t1_img,
            transformlist=tx['fwdtransforms'],
            interpolator='bSpline'
        )

        # 7. 保存结果
        print(f"{log_prefix} SAVING 结果...")
        ants.image_write(t1_warped, output_path)
        
        print(f"{log_prefix} SUCCESS.")
        return (patient_id, 'success', None)

    except Exception as e:
        error_msg = f"发生意外错误: {e}"
        print(f"{log_prefix} FAILED: {error_msg}")
        return (patient_id, 'failed', error_msg)


def register_t1_to_mra_batch(base_dir, overwrite=False, num_workers=4):
    """
    使用多进程并行处理所有患者。
    """
    
    # 查找所有患者文件夹
    try:
        patient_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
        patient_folders = sorted(patient_folders)
    except FileNotFoundError:
        print(f"错误：基础目录 '{base_dir}' 不存在。")
        return

    if not patient_folders:
        print(f"错误：在 '{base_dir}' 中未找到任何患者子文件夹。")
        return

    print(f"总共找到 {len(patient_folders)} 个患者文件夹。")
    print(f"启动 {num_workers} 个并行工作进程...")
    print(f"Overwrite 模式: {'ON' if overwrite else 'OFF'}")

    # 创建任务列表
    tasks = [(patient_dir, overwrite) for patient_dir in patient_folders]

    total_start_time = time.time()
    
    # 使用 multiprocessing.Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(process_patient, tasks)

    total_end_time = time.time()

    # --- 汇总结果 ---
    print("\n" + "="*30 + " 批处理摘要 " + "="*30)
    print(f"总耗时: {(total_end_time - total_start_time) / 60:.2f} 分钟")

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

    print(f"成功处理: {success_count}")
    print(f"跳过 (已存在): {skipped_count}")
    print(f"失败: {len(failed_list)}")
    
    if failed_list:
        print("\n--- 失败详情 ---")
        for patient_id, error_msg in failed_list:
            print(f"  - {patient_id}: {error_msg}")
    print("="*74)


# --- 主程序入口 ---
if __name__ == "__main__":
    
    # 2. 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="批量将 T1 图像配准到 MRA 空间")
    
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

    # 3. 解析参数
    args = parser.parse_args()

    # 4. 验证 base_dir 是否存在
    if not os.path.isdir(args.base_dir):
        print("="*50)
        print(f"错误：指定的 --base_dir 路径不存在或不是一个目录！")
        print(f"路径: {args.base_dir}")
        print("="*50)
    else:
        # 在 Windows 和 macOS 上, multiprocessing 默认使用 'spawn' 模式
        multiprocessing.set_start_method('fork', force=True)
        
        # 5. 使用解析到的参数调用主函数
        register_t1_to_mra_batch(args.base_dir, args.overwrite, args.num_workers)