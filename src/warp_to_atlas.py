import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk

from layers import DeformationFieldComposer, SpatialTransformer
from build_atlas_inference import load_model


def _clip_and_normalize(image: torch.Tensor, pct: float = 0.998) -> torch.Tensor:
    pct = pct if pct <= 1 else pct / 100.0
    img_min = image.min()
    with torch.no_grad():
        try:
            img_max = torch.quantile(image, pct)
        except RuntimeError:
            img_max_np = np.percentile(image.cpu().numpy(), pct * 100.0)
            img_max = torch.tensor(img_max_np, dtype=image.dtype, device=image.device)
    denom = (img_max - img_min).clamp_min(1e-6)
    return torch.clamp((image - img_min) / denom, 0.0, 1.0)


def _compute_padding(shape: Tuple[int, int, int], divisor: int) -> Dict[str, Tuple[int, int]]:
    pads = {}
    labels = ('d', 'h', 'w')
    for dim, label in zip(shape, labels):
        pad_total = (divisor - dim % divisor) % divisor
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pads[label] = (pad_before, pad_after)
    return pads


def _pad_tensor(tensor: torch.Tensor,
                padding: Dict[str, Tuple[int, int]],
                value: float = 0.0) -> torch.Tensor:
    if all(sum(pad) == 0 for pad in padding.values()):
        return tensor
    d_before, d_after = padding['d']
    h_before, h_after = padding['h']
    w_before, w_after = padding['w']
    pad_args = [w_before, w_after, h_before, h_after, d_before, d_after]
    original_shape = tensor.shape
    if tensor.dim() == 5:
        padded = F.pad(tensor, pad_args, mode='constant', value=value)
    elif tensor.dim() == 6:
        b, g = tensor.shape[:2]
        flat = tensor.view(b * g, *tensor.shape[2:])
        padded = F.pad(flat, pad_args, mode='constant', value=value)
        padded = padded.view(b, g, *padded.shape[1:])
    else:
        raise ValueError(f'Unsupported tensor dim {tensor.dim()} for padding, expected 5 or 6 (got shape {original_shape}).')
    return padded


def _crop_tensor(tensor: torch.Tensor, padding: Dict[str, Tuple[int, int]]) -> torch.Tensor:
    d_before, d_after = padding['d']
    h_before, h_after = padding['h']
    w_before, w_after = padding['w']
    d_end = tensor.shape[-3] - d_after if d_after > 0 else tensor.shape[-3]
    h_end = tensor.shape[-2] - h_after if h_after > 0 else tensor.shape[-2]
    w_end = tensor.shape[-1] - w_after if w_after > 0 else tensor.shape[-1]
    return tensor[..., d_before:d_end, h_before:h_end, w_before:w_end]


def _read_volume_sitk(path: str) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)  # (z, y, x)
    tensor = torch.from_numpy(array).float()
    metadata = {
        'spacing': image.GetSpacing(),
        'direction': image.GetDirection(),
        'origin': image.GetOrigin(),
    }
    return tensor, metadata


def _load_volume(path: str, normalize: bool = True) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
    volume, metadata = _read_volume_sitk(path)
    if normalize:
        volume = _clip_and_normalize(volume)
    return volume, metadata


def _save_sitk_image(volume: torch.Tensor,
                     metadata: Dict[str, Tuple],
                     path: str,
                     dtype=np.float32,
                     is_vector: bool = False) -> None:
    array = volume.detach().cpu().numpy().astype(dtype, copy=False)
    sitk_image = sitk.GetImageFromArray(array, isVector=is_vector)
    sitk_image.SetSpacing(metadata['spacing'])
    sitk_image.SetDirection(metadata['direction'])
    sitk_image.SetOrigin(metadata['origin'])
    sitk.WriteImage(sitk_image, path)


def warp_image_and_segmentation_to_atlas(model_path: str,
                                         moving_image_path: str,
                                         moving_segmentation_path: str,
                                         atlas_image_path: str,
                                         output_dir: str,
                                         divisor: int = 16,
                                         device: Optional[str] = None) -> Dict[str, str]:
    """
    Warp a subject image and segmentation mask into an atlas space using a pretrained MultiMorph model.

    Args:
        model_path: path to the pretrained weights.
        moving_image_path: path to the subject image to warp.
        moving_segmentation_path: path to the subject segmentation mask.
        atlas_image_path: path to the atlas image (target space).
        output_dir: directory where warped outputs will be saved.
        divisor: spatial dimensions are padded to be divisible by this value.
        device: torch device string. Defaults to CUDA when available, otherwise CPU.

    Returns:
        Dictionary with file paths to the warped image, segmentation, and deformation field.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch_device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    moving_image, _moving_metadata = _load_volume(moving_image_path, normalize=True)
    atlas_image, atlas_metadata = _load_volume(atlas_image_path, normalize=True)
    moving_seg, _ = _load_volume(moving_segmentation_path, normalize=False)

    if moving_image.shape != atlas_image.shape:
        raise ValueError('Moving image and atlas image must share the same spatial shape before padding.')
    if moving_seg.shape != moving_image.shape:
        raise ValueError('Segmentation mask must have the same spatial shape as the moving image.')

    image_stack = torch.stack([moving_image, atlas_image], dim=0).unsqueeze(1)
    seg_tensor = moving_seg.unsqueeze(0).unsqueeze(0)

    padding = _compute_padding(image_stack.shape[-3:], divisor)
    image_stack = _pad_tensor(image_stack, padding, value=0.0)
    seg_tensor = _pad_tensor(seg_tensor, padding, value=0.0)

    image_batch = image_stack.unsqueeze(0)
    img_size = list(image_batch.shape[-3:])

    model = load_model(model_path, img_size=img_size, output_inverse_field=True)
    model = model.to(torch_device)
    model.eval()

    image_batch = image_batch.to(torch_device)
    seg_tensor = seg_tensor.to(torch_device)

    with torch.no_grad():
        forward_fields, inverse_fields = model(image_batch)

    moving_field = forward_fields[:, 0:1, ...]
    atlas_inverse_field = inverse_fields[:, 1:2, ...]

    composer = DeformationFieldComposer(tuple(img_size))
    composed_field = composer([moving_field, atlas_inverse_field])[:, 0, ...]

    trf_lin = SpatialTransformer(tuple(img_size), mode='bilinear').to(torch_device)
    trf_nearest = SpatialTransformer(tuple(img_size), mode='nearest').to(torch_device)

    moving_image_tensor = image_batch[:, 0, ...]
    warped_image = trf_lin(moving_image_tensor, composed_field)
    warped_seg = trf_nearest(seg_tensor, composed_field)

    warped_image = _crop_tensor(warped_image.cpu(), padding).squeeze(0)
    warped_seg = torch.round(_crop_tensor(warped_seg.cpu(), padding)).squeeze(0)
    composed_field = _crop_tensor(composed_field.cpu(), padding).squeeze(0)

    warped_image_path = os.path.join(output_dir, 'warped_image_to_atlas.nii.gz')
    warped_seg_path = os.path.join(output_dir, 'warped_segmentation_to_atlas.nii.gz')
    deformation_field_path = os.path.join(output_dir, 'moving_to_atlas_displacement.nii.gz')

    _save_sitk_image(warped_image.squeeze(0), atlas_metadata, warped_image_path, dtype=np.float32)
    _save_sitk_image(warped_seg.squeeze(0), atlas_metadata, warped_seg_path, dtype=np.int16)

    deformation_tensor = composed_field.permute(1, 2, 3, 0)
    _save_sitk_image(deformation_tensor, atlas_metadata, deformation_field_path, dtype=np.float32, is_vector=True)

    return {
        'warped_image': warped_image_path,
        'warped_segmentation': warped_seg_path,
        'deformation_field': deformation_field_path,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Warp a subject image and segmentation into an atlas space.')
    parser.add_argument('--model_path', required=True, help='Path to pretrained MultiMorph weights.')
    parser.add_argument('--moving_image', required=True, help='Path to the subject image to be warped.')
    parser.add_argument('--moving_segmentation', required=True, help='Path to the subject segmentation mask.')
    parser.add_argument('--atlas_image', required=True, help='Path to the atlas image defining the target space.')
    parser.add_argument('--output_dir', required=True, help='Directory to store warped outputs.')
    parser.add_argument('--divisor', type=int, default=16, help='Spatial padding divisor (default: 16).')
    parser.add_argument('--device', default=None, help='Torch device string (e.g., "cuda:0" or "cpu").')

    args = parser.parse_args()

    paths = warp_image_and_segmentation_to_atlas(
        model_path=args.model_path,
        moving_image_path=args.moving_image,
        moving_segmentation_path=args.moving_segmentation,
        atlas_image_path=args.atlas_image,
        output_dir=args.output_dir,
        divisor=args.divisor,
        device=args.device,
    )

    print('Warped outputs saved:')
    for key, value in paths.items():
        print(f'  {key}: {value}')


if __name__ == '__main__':
    main()
