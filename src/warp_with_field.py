import os
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'

from typing import Dict, Tuple, Optional

import argparse
import numpy as np
import torch
import SimpleITK as sitk

from layers import SpatialTransformer


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


def _read_volume_sitk(path: str) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    tensor = torch.from_numpy(array).float()
    metadata = {
        'spacing': image.GetSpacing(),
        'direction': image.GetDirection(),
        'origin': image.GetOrigin(),
    }
    return tensor, metadata


def _load_volume(path: str, normalize: bool = False, pct: float = 0.998) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
    volume, metadata = _read_volume_sitk(path)
    if normalize:
        volume = _clip_and_normalize(volume, pct=pct)
    return volume, metadata


def _save_sitk_image(volume: torch.Tensor,
                     metadata: Dict[str, Tuple],
                     path: str,
                     dtype=np.float32) -> None:
    array = volume.detach().cpu().numpy().astype(dtype, copy=False)
    sitk_image = sitk.GetImageFromArray(array)
    sitk_image.SetSpacing(metadata['spacing'])
    sitk_image.SetDirection(metadata['direction'])
    sitk_image.SetOrigin(metadata['origin'])
    sitk.WriteImage(sitk_image, path)


def _load_deformation_field(path: str) -> Tuple[torch.Tensor, Dict[str, Tuple]]:
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    if array.ndim != 4 or array.shape[-1] not in (2, 3):
        raise ValueError(
            f'Deformation field "{path}" must have shape (D, H, W, C) with C in {{2, 3}}, got {tuple(array.shape)}.'
        )
    tensor = torch.from_numpy(array).float().permute(3, 0, 1, 2)
    metadata = {
        'spacing': image.GetSpacing(),
        'direction': image.GetDirection(),
        'origin': image.GetOrigin(),
    }
    return tensor, metadata


def warp_image_with_field(moving_image_path: str,
                          deformation_field_path: str,
                          reference_image_path: str,
                          output_path: str,
                          interpolation: str = 'bilinear',
                          normalize: bool = False,
                          normalize_pct: float = 0.998,
                          device: Optional[str] = None) -> str:
    interpolation = interpolation.lower()
    if interpolation not in {'bilinear', 'nearest'}:
        raise ValueError(f'Unsupported interpolation mode "{interpolation}". Expected "bilinear" or "nearest".')

    moving_volume, _ = _load_volume(moving_image_path, normalize=normalize, pct=normalize_pct)
    reference_volume, reference_metadata = _load_volume(reference_image_path, normalize=False)
    deformation_field, _ = _load_deformation_field(deformation_field_path)

    if deformation_field.shape[0] != len(reference_volume.shape):
        raise ValueError(
            f'Deformation field has {deformation_field.shape[0]} spatial channels but volume dimensionality is {len(reference_volume.shape)}.'
        )

    if tuple(moving_volume.shape) != tuple(deformation_field.shape[1:]):
        raise ValueError(
            f'Moving image spatial shape {tuple(moving_volume.shape)} does not match deformation field spatial shape {tuple(deformation_field.shape[1:])}.'
        )

    if tuple(reference_volume.shape) != tuple(deformation_field.shape[1:]):
        raise ValueError(
            f'Reference image spatial shape {tuple(reference_volume.shape)} does not match deformation field spatial shape {tuple(deformation_field.shape[1:])}.'
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    torch_device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    moving_batch = moving_volume.unsqueeze(0).unsqueeze(0).to(torch_device)
    field_batch = deformation_field.unsqueeze(0).to(torch_device)

    transformer = SpatialTransformer(tuple(reference_volume.shape), mode=interpolation).to(torch_device)
    with torch.no_grad():
        warped = transformer(moving_batch, field_batch)

    warped = warped.squeeze(0).squeeze(0).cpu()
    if normalize:
        warped = warped.clamp(0.0, 1.0)

    _save_sitk_image(warped, reference_metadata, output_path)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Warp an image using a precomputed deformation field.')
    parser.add_argument('--moving_image', required=True, help='Path to the moving image to be warped.')
    parser.add_argument('--deformation_field', required=True, help='Path to the deformation field (vector image).')
    parser.add_argument('--reference_image', required=True, help='Reference image defining the target space.')
    parser.add_argument('--output_path', required=True, help='Output path for the warped image.')
    parser.add_argument('--interpolation', default='bilinear', choices=['bilinear', 'nearest'],
                        help='Interpolation mode to use for warping (default: bilinear).')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize the moving image to [0, 1] before warping.')
    parser.add_argument('--normalize_pct', type=float, default=0.998,
                        help='Upper percentile for normalization when --normalize is set (default: 0.998).')
    parser.add_argument('--device', default=None,
                        help='Torch device string, e.g., "cuda:0" or "cpu". Default uses CUDA if available.')

    args = parser.parse_args()

    output_path = warp_image_with_field(
        moving_image_path=args.moving_image,
        deformation_field_path=args.deformation_field,
        reference_image_path=args.reference_image,
        output_path=args.output_path,
        interpolation=args.interpolation,
        normalize=args.normalize,
        normalize_pct=args.normalize_pct,
        device=args.device,
    )

    print(f'Warped image saved to {output_path}')


if __name__ == '__main__':
    main()