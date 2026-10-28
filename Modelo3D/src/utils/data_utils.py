import numpy as np
import torch
import torchio as tio
import torch.nn.functional as F
import math


def roi_crop_3d(image):
    # 3d image as input
    # Mask of non-black pixels (assuming image has a single channel).
    mask = image > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1, z0:z1]

    padded_crop = tio.CropOrPad(
        np.max(cropped.shape))(cropped.copy()[None])

    # padded_crop = np.transpose(padded_crop, (1, 2, 3, 0))

    return padded_crop

def roi_crop(image):
    # Mask of non-black pixels (assuming image has a single channel).
    mask = image > 0

    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)

    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top

    # Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1]
    
    cropped = cropped[np.newaxis, :, :, np.newaxis]

    padded_crop = tio.CropOrPad(
        (np.max(cropped.shape), np.max(cropped.shape), 1))(cropped.copy())
    # padded_crop = tio.RescaleIntensity(out_min_max=(0, 1))(padded_crop)
    padded_crop = padded_crop.squeeze(-1)

    # padded_crop = np.transpose(padded_crop, (1, 2, 3, 0))

    return padded_crop

def crop_minimally_keep_ratio(image, target_size):
    original_height, original_width = image.shape[-2:]
    target_height, target_width = target_size

    # Calculate the aspect ratios
    aspect_ratio_original = original_width / original_height
    aspect_ratio_target = target_width / target_height

    # Determine whether to crop width or height
    if aspect_ratio_original > aspect_ratio_target:  # Crop width
        new_width_cropped = original_height * aspect_ratio_target
        left_boundary = int((original_width - new_width_cropped) / 2)
        right_boundary = int(original_width - left_boundary)
        cropped_image = image[:, left_boundary:right_boundary]

    else:  # Crop height
        new_height_cropped = original_width / aspect_ratio_target
        top_boundary = int((original_height - new_height_cropped) / 2)
        bottom_boundary = int(original_height - top_boundary)
        cropped_image = image[top_boundary:bottom_boundary, :]

    return cropped_image


# data transforms

def get_3d_image_transform(resolution, random_crop: bool = False, random_flip: bool = False, random_affine: bool = False):
    # Image sizes  PET & MRI        Dataset {113, 137, 113}
    img_transforms = [
        tio.RescaleIntensity(out_min_max=(0, 1))
    ]

    if resolution:
        CropOrPad = tio.CropOrPad(resolution)
        img_transforms.append(CropOrPad)
        # Rescale = montrans.Resize(resolution)
        # img_transforms.append(Rescale)

    if random_flip:
        Flip = tio.RandomFlip(axes = (0,1,2), flip_probability=0.5)
        img_transforms.append(Flip)
    
    if random_affine:
        randomAffineWithRot = tio.RandomAffine(
            scales=0.05,
            degrees=90,  # +-90 degree in each dimension
            translation=8,  # +-8 pixels offset in each dimension.
            image_interpolation="linear",
            default_pad_value="otsu",
            p=0.5,
        )
        img_transforms.append(randomAffineWithRot)

    img_transform = tio.Compose(img_transforms)
    return img_transform

def rescale_intensity(img):
    img = img[np.newaxis, :, :, np.newaxis]
    img = tio.RescaleIntensity(out_min_max=(0, 1))(img)
    img = img.squeeze(0)
    img = img.squeeze(-1)
    return img

def window_ncct(img, window_center=40.0, window_width=80.0):
    """
    Brain window estándar para Tomografía Computarizada (NCCT):
    Rango [0, 80] HU. Escala el parénquima cerebral (materia gris/blanca y lesión isquémica)
    al rango [0, 1] completo, eliminando la dominancia del hueso y aire.
    """
    min_hu = window_center - window_width / 2.0  # 0.0 HU
    max_hu = window_center + window_width / 2.0  # 80.0 HU
    img_clipped = np.clip(img, min_hu, max_hu)
    return (img_clipped - min_hu) / (max_hu - min_hu)

def robust_scale_dwi(img, max_val=2000.0):
    """
    Normalización constante para DWI basada en escala física fija.
    Mantiene la consistencia de intensidad exacta entre pacientes y lesiones.
    - Tejido cerebral normal (~200-700) -> ~0.10-0.35
    - Lesión isquémica agudizada (~1000-1800) -> ~0.50 - 0.90

    NOTA: max_val se subió de 1000.0 a 2000.0. Con 1000.0, ~57% de los
    voxeles de lesión (que en datos reales llegan hasta ~1800, con el
    volumen completo hasta ~2100) quedaban saturados en el mismo valor
    1.0, borrando la variación de severidad de la isquemia (justamente
    la señal clínica más relevante del DWI). 2000.0 cubre el rango real
    observado sin recortar la lesión.
    """
    img_clipped = np.clip(img, 0.0, max_val)
    return img_clipped / max_val

def unscale_dwi(img, max_val=2000.0):
    """
    Desnormaliza la imagen sintética DWI del rango [0, 1] a sus unidades NIfTI físicas originales [0, 2000].
    Debe mantenerse en el mismo valor de max_val que robust_scale_dwi.
    """
    return np.clip(img, 0.0, 1.0) * max_val

def rescale_intensity_3D(img):
    img = img[np.newaxis, :, :, :]
    img = tio.RescaleIntensity(out_min_max=(0, 1))(img)
    img = img.squeeze(0)
    return img

def CropOrPad_3D(img, resolution):
    img = img[np.newaxis, :, :, :]
    img = tio.CropOrPad(resolution)(img)
    img = img.squeeze(0)
    return img

def resample_and_reshape(img, new_spacing=(1.5, 1.5, 1.5), target_shape=(113, 137, 113)):
    # resample to 1.5mm spacing using monai
    resampled_mri = tio.Resample(new_spacing, image_interpolation = 'linear')(img[np.newaxis])
    rescaled_mri = tio.CropOrPad(target_shape)(resampled_mri)
    rescaled_mri = rescaled_mri.squeeze(0)
    return rescaled_mri

def process_tabular_data(tabular_data):
    """
    Procesa los datos clínicos/tabulares convirtiendo valores NaN a 0.0
    y devolviendo un tensor PyTorch con las características completas (ej: 27 para ACV).
    """
    tabular_data = np.array(tabular_data, dtype=np.float32)
    tabular_data = np.nan_to_num(tabular_data, copy=False)
    return torch.from_numpy(tabular_data)

def data_minimal_process(resolution, input_scan, data_transform):
    if not np.any(input_scan):
        # return the image of same size with all zeros
        input_scan = np.zeros_like(input_scan)

    input_scan = np.array(input_scan).astype(np.float32)

    # if input_scan.shape[0] / input_scan.shape[1] != resolution[0] / resolution[1]:
    #     input_scan = crop_minimally_keep_ratio(input_scan, resolution)

    input_scan = data_transform(input_scan)

    return input_scan

def pad_slices(slices_list, num_slices_needed, at_beginning):
    """
    Helper function to pad the given slices_list with zeros.
    """
    # Create the zero slices
    zero_slices = [np.zeros_like(slices_list[0]) for _ in range(num_slices_needed)]
    
    # Add the zero slices to the given list
    if at_beginning:
        return zero_slices + slices_list
    else:
        return slices_list + zero_slices


def get_neighboring_slices(output_dim, direction, slice_num, scan):

    scan_shape = scan.shape
    
    # Number of slices before and after the current slice
    half_dim = (output_dim - 1) // 2

    # Handle edge cases: pad with zeros if necessary
    start_idx = max(0, slice_num - half_dim)      
            
    if direction == 'coronal':
        end_idx = min(scan_shape[1], slice_num + half_dim + 1)
        neighboring_slices = [scan[:, idx, :] for idx in range(start_idx, end_idx)]
        padding_before = (output_dim - len(neighboring_slices)) if start_idx == 0 else 0
        padding_after = (output_dim - len(neighboring_slices) - padding_before) if end_idx == scan_shape[1] else 0
            
    elif direction == 'sagittal':
        end_idx = min(scan_shape[0], slice_num + half_dim + 1)
        neighboring_slices = [scan[idx, :, :] for idx in range(start_idx, end_idx)]
        padding_before = (output_dim - len(neighboring_slices)) if start_idx == 0 else 0
        padding_after = (output_dim - len(neighboring_slices) - padding_before) if end_idx == scan_shape[0] else 0

    elif direction == 'axial':
        end_idx = min(scan_shape[2], slice_num + half_dim + 1)
        neighboring_slices = [scan[:, :, idx] for idx in range(start_idx, end_idx)]
        padding_before = (output_dim - len(neighboring_slices)) if start_idx == 0 else 0
        padding_after = (output_dim - len(neighboring_slices) - padding_before) if end_idx == scan_shape[2] else 0
    
    
    if padding_before > 0:
        neighboring_slices = pad_slices(neighboring_slices, padding_before, at_beginning=True)
    
    if padding_after > 0:
        neighboring_slices = pad_slices(neighboring_slices, padding_after, at_beginning=False)

    # Stack the slices to form the image with output_dim channels      
    _output_data = np.stack(neighboring_slices, axis=0)

    return _output_data


def multislice_data_minimal_process(output_dim, resolution, input_scan, data_transform):
    processed_scan = np.zeros((output_dim, input_scan.shape[1], input_scan.shape[2])).astype(np.float32)

    for i in range(input_scan.shape[0]):
        _input_data = input_scan[i, :, :]

        _input_data = data_minimal_process(resolution, _input_data, data_transform)

        processed_scan[i, :, :] = _input_data
    
    return processed_scan

def gaussian_weight(x, b, c):
    return math.exp(-((x - b) ** 2) / (2 * c ** 2))


def reconstruct_scan_from_2_5D_slices(sets_of_slices):
    """
    Reconstruct a 3D scan from sets of slices.

    Parameters:
    - sets_of_slices: a List of tensor sets. Each tensor set has shape (B, C, H, W).

    Returns:
    - A reconstructed 3D tensor scan of shape (B, N, H, W).
    """

    # Determine the number of slices in each set and validate it's odd
    B, slices_per_set, h, w = sets_of_slices[0].shape
    slice_shape = (h, w)
    if slices_per_set % 2 == 0:
        raise ValueError("Number of slices per set should be odd.")

    # Compute the offset due to extra slices on each side of the set
    offset = slices_per_set // 2

    # Total number of slices in the final virtual 3D scan
    n = len(sets_of_slices) + 2 * offset

    # Initialize the 3D tensor and the counter tensor
    scan_3D = torch.zeros(B, n, *slice_shape).to(sets_of_slices[0].device)
    count = torch.zeros(B, n, *slice_shape).to(sets_of_slices[0].device)


    c = offset / 3.0  # You can adjust this for desired spread
    
    # weights = [gaussian_weight(i, offset, c) for i in range(slices_per_set)] # gaussian
    weights = [1 - abs(offset - i) / float(offset) for i in range(slices_per_set)] # linear
    
    weights_tensor = torch.tensor(weights).to(sets_of_slices[0].device)
    
    for i, set_of_slices in enumerate(sets_of_slices):
        for j in range(slices_per_set):
            # weight the slice by its position in the set
            idx = i + j
            scan_3D[:, idx] += set_of_slices[:, j] * weights_tensor[j]
            count[:, idx] += weights_tensor[j]
            # count[:, idx] += 1

    # Get the average of overlapping slices
    scan_3D /= count.clamp(min=1)  # clamp to prevent division by zero

    # Extract slices from offset to n-offset and append to a list
    result_slices = [scan_3D[:, i] for i in range(offset, n - offset)]

    return result_slices

# ---------------------------------------------------------------------------
# 3D volume helpers (added for the 3D architecture)
# ---------------------------------------------------------------------------
# Convention used everywhere in the 3D code base:
#   * NIfTI / H5 volumes are stored as (H, W, D)  -> "HWD"
#   * Tensors fed to the network are            (C, D, H, W) -> "CDHW"
#     (D = axial slice axis. Conv3d treats the first spatial axis as depth, and the
#      UNet only down/up-samples H and W, so the thick-slice axis D is preserved.)

def hwd_to_dhw(vol):
    """(H, W, D) -> (D, H, W)"""
    return np.ascontiguousarray(np.transpose(vol, (2, 0, 1)))


def dhw_to_hwd(vol):
    """(D, H, W) -> (H, W, D)"""
    return np.ascontiguousarray(np.transpose(vol, (1, 2, 0)))


def _split_ceil_floor(n):
    """torchio convention: ini = ceil(n/2), fin = floor(n/2)."""
    return int(math.ceil(n / 2)), int(math.floor(n / 2))


def crop_or_pad_3d_np(vol, target_shape, value=0.0):
    """
    Centered crop / zero-pad of a 3D numpy array to ``target_shape``. Pure numpy and
    numerically identical to ``torchio.CropOrPad(target_shape)`` (same centering rule),
    but without depending on the torchio version.
    """
    assert vol.ndim == 3, vol.shape
    out = vol
    for axis, (cur, tgt) in enumerate(zip(vol.shape, target_shape)):
        tgt = int(tgt)
        if cur == tgt:
            continue
        if cur > tgt:
            ini, fin = _split_ceil_floor(cur - tgt)
            sl = [slice(None)] * 3
            sl[axis] = slice(ini, cur - fin)
            out = out[tuple(sl)]
        else:
            ini, fin = _split_ceil_floor(tgt - cur)
            pad = [(0, 0)] * 3
            pad[axis] = (ini, fin)
            out = np.pad(out, pad, mode='constant', constant_values=value)
    return np.ascontiguousarray(out)


def resize_inplane_np(vol_hwd, size_hw, antialias=True):
    """
    Bilinear resize of the in-plane (H, W) axes of an (H, W, D) array, slice by slice.
    Equivalent to the per-slice ``T.Resize`` of the 2.5D pipeline; depth is untouched.
    """
    h, w, d = vol_hwd.shape
    if (h, w) == tuple(size_hw):
        return vol_hwd
    t = torch.from_numpy(np.ascontiguousarray(vol_hwd)).permute(2, 0, 1).unsqueeze(1).float()  # (D,1,H,W)
    use_aa = antialias and (size_hw[0] < h or size_hw[1] < w)
    t = F.interpolate(t, size=tuple(size_hw), mode='bilinear', align_corners=False, antialias=use_aa)
    return t.squeeze(1).permute(1, 2, 0).contiguous().numpy()


def to_network_volume(vol_hwd, volume_size_hwd, inplane_resize=True, is_mask=False):
    """
    (H, W, D) float array -> (D, H, W) float32 array with the requested size.

    :param volume_size_hwd: target (H, W, D).
    :param inplane_resize: if True, H/W are resized (as the 2.5D pipeline did) and only
        D is cropped/padded; if False, H, W and D are all center cropped / zero padded.
    :param is_mask: binary mask -> resized with bilinear then re-binarised (> 0.5).
    """
    vol = np.asarray(vol_hwd, dtype=np.float32)
    H, W, D = [int(v) for v in volume_size_hwd]
    if inplane_resize:
        vol = resize_inplane_np(vol, (H, W))
        vol = crop_or_pad_3d_np(vol, (H, W, D))
    else:
        vol = crop_or_pad_3d_np(vol, (H, W, D))
    if is_mask:
        vol = (vol > 0.5).astype(np.float32)
    return hwd_to_dhw(vol.astype(np.float32))
