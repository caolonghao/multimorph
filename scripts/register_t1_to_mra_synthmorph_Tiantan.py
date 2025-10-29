#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
from pathlib import Path
import argparse

def register_t1_to_mra(datashare_dir, overwrite=False):
    """
    遍历datashare目录中的所有子文件夹，将T1.nii.gz配准到同一文件夹中的MRA.nii.gz
    
    Args:
        datashare_dir (str): datashare目录的路径
        overwrite (bool): 是否覆盖已存在的输出文件（用于断点续传）
    """
    datashare_path = Path(datashare_dir)
    
    # 检查datashare目录是否存在
    if not datashare_path.exists():
        print(f"错误: 目录 {datashare_dir} 不存在")
        return
    
    # 遍历所有子文件夹
    subject_dirs = [d for d in datashare_path.iterdir() if d.is_dir()]
    subject_dirs.sort(key=lambda x: x.name)
    for subject_dir in subject_dirs:
            t1_path = subject_dir / "Resampled" / "T1_warped_resampled.nii.gz"
            mra_path = subject_dir / "Resampled" / "MRA_resampled.nii.gz"

            # 检查T1.nii.gz和MRA.nii.gz是否存在
            if t1_path.exists() and mra_path.exists():
                # 定义输出文件路径
                t1_moved_path = subject_dir / "Resampled" / "T1_synthmorph_resampled.nii.gz"

                # 断点续传：若输出已存在且未指定覆盖，则跳过
                if t1_moved_path.exists() and not overwrite:
                    try:
                        size_ok = t1_moved_path.stat().st_size > 0
                    except Exception:
                        size_ok = True  # 无法获取大小则按存在处理
                    if size_ok:
                        print(f"跳过 {subject_dir.name}: 输出已存在。使用 --overwrite 可覆盖。")
                        continue

                # 构建命令
                cmd = [
                    "./synthmorph", "register",
                    "-o", str(t1_moved_path),
                    str(t1_path),
                    str(mra_path)
                ]

                print(f"正在处理: {subject_dir.name}")
                print(f"命令: {' '.join(cmd)}")

                try:
                    # 执行命令
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    print(f"成功完成: {subject_dir.name}")
                    if result.stdout:
                        print(f"输出: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"错误处理 {subject_dir.name}: {e}")
                    if e.stderr:
                        print(f"错误信息: {e.stderr}")
                except FileNotFoundError:
                    print("错误: 找不到 synthmorph 命令，请确保它在当前目录中或在PATH中")
                    return
                print("-" * 50)
            else:
                print(f"跳过 {subject_dir.name}: 缺少 T1.nii.gz 或 MRA.nii.gz")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将T1图像配准到MRA图像")
    parser.add_argument("-d", "--datashare_dir", required=True, help="datashare目录路径")
    parser.add_argument("-O", "--overwrite", action="store_true", help="是否覆盖已存在的输出文件")
    args = parser.parse_args()
    
    # 运行配准函数
    register_t1_to_mra(args.datashare_dir, overwrite=args.overwrite)
