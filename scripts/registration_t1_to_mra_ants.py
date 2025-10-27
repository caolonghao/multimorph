import ants
import os
import time

def register_t1_to_mra(base_dir):
    """
    遍历指定目录下的所有子文件夹，将 T1.nii.gz 配准到 MRA.nii.gz。

    参数:
    base_dir (str): 包含所有患者子文件夹的根目录路径 (例如 './MRA_label_v2')。
    """
    print(f"开始处理目录: {base_dir}")
    
    # 获取所有子文件夹（代表每个患者）
    subject_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    
    if not subject_folders:
        print("错误：在指定目录下未找到任何子文件夹。")
        return

    for subject_path in subject_folders:
        subject_id = os.path.basename(subject_path)
        print(f"\n--- 正在处理患者: {subject_id} ---")

        # 定义文件路径
        mra_path = os.path.join(subject_path, 'MRA.nii.gz')
        t1_path = os.path.join(subject_path, 'T1.nii.gz')

        # 检查所需文件是否存在
        if not os.path.exists(mra_path):
            print(f"警告: 在 {subject_path} 中未找到 MRA.nii.gz，跳过此患者。")
            continue
        if not os.path.exists(t1_path):
            print(f"警告: 在 {subject_path} 中未找到 T1.nii.gz，跳过此患者。")
            continue

        try:
            # 1. 加载影像
            print("正在加载影像...")
            fixed_img = ants.image_read(mra_path)
            moving_img = ants.image_read(t1_path)

            # --- 2. 执行刚性配准 (Rigid) ---
            output_rigid_path = os.path.join(subject_path, 'T1_registered_to_MRA_rigid.nii.gz')
            if os.path.exists(output_rigid_path):
                print(f"刚性配准结果已存在，跳过: {output_rigid_path}")
            else:
                print("正在执行刚性(Rigid)配准...")
                start_time_rigid = time.time()
                
                # 使用 ants.registration 函数进行配准, verbose=True 以输出详细信息
                # type_of_transform='Rigid' 包括旋转和平移
                reg_rigid = ants.registration(
                    fixed=fixed_img,
                    moving=moving_img,
                    type_of_transform='Rigid',
                    verbose=True
                )
                
                # 从返回的字典中获取配准后的影像
                warped_rigid_img = reg_rigid['warpedmovout']
                
                # 保存配准后的影像
                ants.image_write(warped_rigid_img, output_rigid_path)
                end_time_rigid = time.time()
                print(f"刚性配准完成，结果已保存至: {output_rigid_path}")
                print(f"耗时: {end_time_rigid - start_time_rigid:.2f} 秒")


            # --- 3. 执行仿射配准 (Affine) ---
            output_affine_path = os.path.join(subject_path, 'T1_registered_to_MRA_affine.nii.gz')
            if os.path.exists(output_affine_path):
                print(f"\n仿射配准结果已存在，跳过: {output_affine_path}")
            else:
                print("\n正在执行仿射(Affine)配准...")
                start_time_affine = time.time()

                # type_of_transform='Affine' 包括旋转、平移、缩放和剪切
                reg_affine = ants.registration(
                    fixed=fixed_img,
                    moving=moving_img,
                    type_of_transform='Affine',
                    verbose=True
                )
                
                warped_affine_img = reg_affine['warpedmovout']
                
                # 保存配准后的影像
                ants.image_write(warped_affine_img, output_affine_path)
                end_time_affine = time.time()
                print(f"仿射配准完成，结果已保存至: {output_affine_path}")
                print(f"耗时: {end_time_affine - start_time_affine:.2f} 秒")

        except Exception as e:
            print(f"处理患者 {subject_id} 时发生错误: {e}")

    print("\n--- 所有患者处理完毕 ---")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 设置你的数据根目录
    data_root_directory = './MRA_label_v2'
    register_t1_to_mra(data_root_directory)