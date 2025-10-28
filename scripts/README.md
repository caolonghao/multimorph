对于Precise数据，预处理运行顺序如下：
1. 先运行`register_t1_to_mra_ants_Tiantan.py`，实现原始 T1 到 MRA 的配准。
2. 运行`resample_multimodal_images_Tiantan.py`，将T1、MRA、血管 mask 都降采样到 256x256x128，保存到 `Resampled` 目录下，便于大批量后续快速处理
3. 运行`register_mra_images_Tiantan.py`，将 MRA 配准到某一个特定患者，对 T1 与血管 mask 应用变换，已实现所有模态的影像到某一特定患者空间的配准。
4. 运行`weighted_average_images_Tiantan.py`，将所有患者的 MRA 与血管 mask 进行加权平均，生成混合图像，并生成 csv 文件便于 multimorph 运行。

T1图像变化路线：T1 -> T1_warped_to_MRA -> T1_warped_resampled -> T1_synthmorph_resampled -> T1_registered_to_target -> T1_weighted
