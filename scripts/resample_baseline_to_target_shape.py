import os
import csv
import SimpleITK as sitk
import numpy as np

def resample_image_to_target_shape(image, target_shape=(512, 512, 120), target_spacing=None, interpolator=sitk.sitkLinear):
    """
    将图像重采样到目标形状
    
    Args:
        image: SimpleITK图像对象
        target_shape: 目标形状 (width, height, depth)
        target_spacing: 目标spacing (x, y, z)，如果指定则先重采样到此spacing
        interpolator: 插值方法
    
    Returns:
        重采样后的SimpleITK图像对象
    """
    # 获取原始图像信息
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()
    
    # 如果指定了target spacing，则先重采样到target spacing
    if target_spacing is not None:
        # 创建临时重采样器，将图像重采样到目标spacing
        temp_resampler = sitk.ResampleImageFilter()
        temp_resampler.SetOutputSpacing(target_spacing)
        temp_resampler.SetOutputOrigin(original_origin)
        temp_resampler.SetOutputDirection(original_direction)
        temp_resampler.SetInterpolator(interpolator)
        temp_resampler.SetDefaultPixelValue(0)
        
        # 计算临时图像的尺寸以保持原始物理尺寸
        temp_size = [
            int(original_size[0] * original_spacing[0] / target_spacing[0]),
            int(original_size[1] * original_spacing[1] / target_spacing[1]),
            int(original_size[2] * original_spacing[2] / target_spacing[2])
        ]
        temp_resampler.SetSize(temp_size)
        
        # 执行第一次重采样到target spacing
        temp_image = temp_resampler.Execute(image)
        
        # 然后从temp_image重采样到目标形状
        temp_size = temp_image.GetSize()
        temp_spacing = temp_image.GetSpacing()
        temp_origin = temp_image.GetOrigin()
        temp_direction = temp_image.GetDirection()
        
        # 计算新的spacing以达到目标shape
        new_spacing = [
            temp_spacing[0] * temp_size[0] / target_shape[0],
            temp_spacing[1] * temp_size[1] / target_shape[1],
            temp_spacing[2] * temp_size[2] / target_shape[2]
        ]
        
        # 设置最终重采样器
        final_resampler = sitk.ResampleImageFilter()
        final_resampler.SetSize(target_shape)
        final_resampler.SetOutputSpacing(new_spacing)
        final_resampler.SetOutputOrigin(temp_origin)
        final_resampler.SetOutputDirection(temp_direction)
        final_resampler.SetInterpolator(interpolator)
        final_resampler.SetDefaultPixelValue(0)
        
        # 执行第二次重采样到目标shape
        resampled_image = final_resampler.Execute(temp_image)
    else:
        # 如果没有指定target spacing，直接重采样到目标形状（原始逻辑）
        # 计算新的spacing以达到目标shape
        new_spacing = [
            original_spacing[0] * original_size[0] / target_shape[0],
            original_spacing[1] * original_size[1] / target_shape[1],
            original_spacing[2] * original_size[2] / target_shape[2]
        ]
        
        # 设置重采样器
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_shape)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(original_origin)
        resampler.SetOutputDirection(original_direction)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(0)
        
        # 执行重采样
        resampled_image = resampler.Execute(image)
    
    return resampled_image
def normalize_image_to_01(image):
    """
    将图像归一化到0-1范围
    
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
    if max_val - min_val == 0:
        normalized_array = np.zeros_like(image_array)
    else:
        # 归一化到0-1范围
        normalized_array = (image_array - min_val) / (max_val - min_val)
    
    # 创建新的SimpleITK图像
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)  # 复制原图像的元信息
    
    return normalized_image

def weighted_sum_images(image, segmentation, alpha=0.5):
    """
    对图像和分割图像进行加权求和
    
    Args:
        image: SimpleITK图像对象（已归一化）
        segmentation: SimpleITK分割图像对象
        alpha: 加权系数，image的权重
        
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

def resample_baseline_data(
    input_dir,
    output_dir,
    target_shape=(256, 256, 120),
    target_spacing=None,
    alpha=0.5,
    image_type="T1",  # "T1" 或 "MRA"
):
    """
    重采样baseline数据到目标形状
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        target_shape: 目标形状 (width, height, depth)
        target_spacing: 目标spacing (x, y, z)，如果指定则先重采样到此spacing
        alpha: 加权系数，用于图像增强
        image_type: 图像类型，"T1" 或 "MRA"
    """
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储新的文件路径用于更新metadata
    new_pairs = []
    processed_count = 0
    failed_count = 0
    
    # 获取并排序输入目录中的所有子目录
    patient_dirs = []
    for patient_id in os.listdir(input_dir):
        patient_dir = os.path.join(input_dir, patient_id)
        if os.path.isdir(patient_dir):
            # 根据图像类型检查对应的文件
            if image_type == "T1":
                img_filename = "T1_aligned.nii.gz"
            elif image_type == "MRA":
                img_filename = "aligned_mra_resampled.nii.gz"
            else:
                print(f"错误: 不支持的图像类型 {image_type}")
                return
            
            img_full_path = os.path.join(patient_dir, img_filename)
            if os.path.exists(img_full_path):
                patient_dirs.append((patient_id, patient_dir, img_full_path))
    
    # 按患者ID排序
    patient_dirs.sort(key=lambda x: x[0])
    
    # 遍历排序后的子目录
    for patient_id, patient_dir, img_full_path in patient_dirs:
            
        output_patient_dir = os.path.join(output_dir, patient_id)
        os.makedirs(output_patient_dir, exist_ok=True)
        
        # 根据图像类型设置分割文件路径
        if image_type == "T1":
            seg_filename = "aligned_mra_seg_resampled.nii.gz"
        elif image_type == "MRA":
            seg_filename = "aligned_mra_seg_resampled.nii.gz"  # MRA使用相同的分割文件
        else:
            print(f"错误: 不支持的图像类型 {image_type}")
            failed_count += 1
            continue
            
        seg_full_path = os.path.join(patient_dir, seg_filename)
        
        # 检查分割文件是否存在
        seg_exists = os.path.exists(seg_full_path)
        
        missing_files = []
        if not os.path.exists(img_full_path):
            missing_files.append(img_full_path)
        if seg_exists and not os.path.exists(seg_full_path):
            missing_files.append(seg_full_path)
            
        if missing_files:
            print(f"警告: 以下文件缺失，跳过 {patient_id}: {missing_files}")
            failed_count += 1
            continue
            
        try:
            print(f"处理患者: {patient_id}")
            
            # 处理图像
            image = sitk.ReadImage(img_full_path)
            resampled_image = resample_image_to_target_shape(image, target_shape, target_spacing, sitk.sitkLinear)
            normalized_image = normalize_image_to_01(resampled_image)
            
            # 根据图像类型设置输出文件名
            if image_type == "T1":
                output_img_name = "aligned_T1_resampled.nii.gz"
            elif image_type == "MRA":
                output_img_name = "aligned_mra_resampled.nii.gz"
            else:
                print(f"错误: 不支持的图像类型 {image_type}")
                failed_count += 1
                continue
                
            output_img_path = os.path.join(output_patient_dir, output_img_name)
            sitk.WriteImage(resampled_image, output_img_path)
            print(f"保存原始{image_type}影像: {output_img_path}")
            
            # 如果存在分割文件，也处理分割文件
            if seg_exists:
                segmentation = sitk.ReadImage(seg_full_path)
                resampled_segmentation = resample_image_to_target_shape(
                    segmentation, target_shape, target_spacing, sitk.sitkNearestNeighbor
                )
                
                # 根据图像类型设置分割文件输出名
                if image_type == "T1":
                    output_seg_name = "aligned_T1_seg_resampled.nii.gz"
                elif image_type == "MRA":
                    output_seg_name = "aligned_mra_seg_resampled.nii.gz"
                else:
                    print(f"错误: 不支持的图像类型 {image_type}")
                    failed_count += 1
                    continue
                    
                output_seg_path = os.path.join(output_patient_dir, output_seg_name)
                sitk.WriteImage(resampled_segmentation, output_seg_path)
                print(f"保存分割文件: {output_seg_path}")
                
                # 对图像和分割图像进行加权求和
                weighted_sum_image = weighted_sum_images(normalized_image, resampled_segmentation, alpha)
                
                # 根据图像类型设置加权图像输出名
                if image_type == "T1":
                    output_weighted_img_name = "aligned_T1_resampled_weighted.nii.gz"
                elif image_type == "MRA":
                    output_weighted_img_name = "aligned_mra_resampled_weighted.nii.gz"
                else:
                    print(f"错误: 不支持的图像类型 {image_type}")
                    failed_count += 1
                    continue
                    
                output_weighted_img_path = os.path.join(output_patient_dir, output_weighted_img_name)
                sitk.WriteImage(weighted_sum_image, output_weighted_img_path)
                print(f"保存{image_type}加权图像: {output_weighted_img_path}")
            else:
                # 如果没有分割文件，只保存原始图像
                # 根据图像类型设置加权图像输出名
                if image_type == "T1":
                    output_weighted_img_name = "aligned_T1_resampled_weighted.nii.gz"
                elif image_type == "MRA":
                    output_weighted_img_name = "aligned_mra_resampled_weighted.nii.gz"
                else:
                    print(f"错误: 不支持的图像类型 {image_type}")
                    failed_count += 1
                    continue
                    
                output_weighted_img_path = os.path.join(output_patient_dir, output_weighted_img_name)
                sitk.WriteImage(normalized_image, output_weighted_img_path)
                print(f"保存{image_type}加权图像 (仅归一化): {output_weighted_img_path}")
            
            # 使用系统的绝对路径设置元数据字段
            absolute_img_path = os.path.abspath(os.path.join(output_dir, patient_id, output_img_name))
            absolute_seg_path = os.path.abspath(os.path.join(output_dir, patient_id, output_seg_name)) if seg_exists else ""
            
            # 根据图像类型设置元数据字段
            if image_type == "T1":
                absolute_weighted_path = os.path.abspath(os.path.join(output_dir, patient_id, output_weighted_img_name))
                new_pairs.append(
                    {
                        'img_path': absolute_img_path,
                        'segmentation_path': absolute_seg_path,
                        'weighted_T1_img_path': absolute_weighted_path,
                    }
                )
            elif image_type == "MRA":
                absolute_weighted_path = os.path.abspath(os.path.join(output_dir, patient_id, output_weighted_img_name))
                new_pairs.append(
                    {
                        'img_path': absolute_img_path,
                        'segmentation_path': absolute_seg_path,
                        'weighted_mra_img_path': absolute_weighted_path,
                    }
                )
            else:
                print(f"错误: 不支持的图像类型 {image_type}")
                failed_count += 1
                continue
            
            processed_count += 1
            print(f"成功处理: {patient_id} ({processed_count})")
            
        except Exception as e:
            print(f"处理 {patient_id} 时出错: {e}")
            failed_count += 1
            continue
    
    # 创建新的metadata.csv文件
    output_metadata_file = os.path.join(output_dir, "metadata.csv")
    with open(output_metadata_file, 'w', newline='', encoding='utf-8') as csvfile:
        # 根据图像类型设置字段名
        if image_type == "T1":
            fieldnames = ['img_path', 'segmentation_path', 'weighted_T1_img_path']
        elif image_type == "MRA":
            fieldnames = ['img_path', 'segmentation_path', 'weighted_mra_img_path']
        else:
            print(f"错误: 不支持的图像类型 {image_type}")
            return
            
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in new_pairs:
            writer.writerow(item)
    
    print(f"\n重采样完成!")
    print(f"目标形状: {target_shape}")
    print(f"目标spacing: {target_spacing}")
    print(f"输出文件夹: {output_dir}")
    print(f"元数据文件: {output_metadata_file}")
    print(f"成功处理: {processed_count} 对文件")
    print(f"失败: {failed_count} 对文件")
    
    return output_dir, output_metadata_file, processed_count

def check_shape_consistency(metadata_file):
    """
    检查重采样后文件的形状一致性
    """
    print("\n检查文件形状一致性...")
    
    with open(metadata_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        shapes = []
        
        for i, row in enumerate(reader):
            if i >= 5:  # 只检查前5个文件
                break
                
            img_path = row['img_path']
            seg_path = row['segmentation_path']
            
            try:
                # 检查原始影像形状
                image = sitk.ReadImage(img_path)
                img_shape = image.GetSize()
                
                # 检查分割文件形状
                segmentation = sitk.ReadImage(seg_path)
                seg_shape = segmentation.GetSize()
                
                shapes.append((img_path, img_shape, seg_shape))
                print(f"文件: {os.path.basename(img_path)}")
                print(f"  原始影像形状: {img_shape}")
                print(f"  分割文件形状: {seg_shape}")
                print(f"  形状匹配: {img_shape == seg_shape}")
                
            except Exception as e:
                print(f"检查 {img_path} 时出错: {e}")
    
    return shapes

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="重采样baseline数据到目标形状")
    parser.add_argument("--input_dir", type=str, default="multimorph/data/baseline_aligned",
                       help="输入文件夹路径")
    parser.add_argument("--output_dir", type=str, default="multimorph/data/baseline_3d_data_resampled",
                       help="输出文件夹路径")
    parser.add_argument("--target_shape", type=int, nargs=3, default=[256, 256, 128],
                       help="目标形状 (width height depth)")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=None,
                       help="目标spacing (x, y, z)，如果指定则先重采样到此spacing")
    parser.add_argument("--check_consistency", action="store_true",
                       help="处理完成后检查形状一致性")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="加权系数，用于图像增强 (默认: 0.5)")
    parser.add_argument("--image_type", type=str, default="T1", choices=["T1", "MRA"],
                       help="图像类型，T1 或 MRA (默认: T1)")
    
    args = parser.parse_args()
    
    print(f"输入文件夹: {args.input_dir}")
    print(f"输出文件夹: {args.output_dir}")
    print(f"目标形状: {tuple(args.target_shape)}")
    print(f"目标spacing: {args.target_spacing}")
    print(f"图像类型: {args.image_type}")
    
    # 执行重采样
    output_dir, metadata_file, processed_count = resample_baseline_data(
        args.input_dir,
        args.output_dir,
        tuple(args.target_shape),
        args.target_spacing,
        args.alpha,
        args.image_type
    )
    
    # 可选：检查形状一致性
    if args.check_consistency and processed_count > 0:
        check_shape_consistency(metadata_file)
