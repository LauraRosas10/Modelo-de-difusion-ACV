"""
3D dataset for NCCT -> DWI translation (ACV / stroke).

Replaces the two 2.5D datasets of the previous version (``MRI2PET_2_5D_Dataset`` and
``SlicedScanMRI2PETDataset``). Instead of slicing every volume into thousands of
(15 or 3)-channel 2D samples, ONE sample == ONE whole 3D volume (or a 3D patch of it
during training).

Tensor layout returned for the network: (C=1, D, H, W)
    D = axial slice axis (kept at full resolution by the UNet),
    H, W = in-plane axes.

The H5 files produced by ``prepare_pasta_h5.py`` are used as-is (they already store whole
3D volumes in (H, W, D) order), so no data re-preparation is needed.
"""
import logging

import h5py
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils.data_utils import (
    window_ncct,
    robust_scale_dwi,
    process_tabular_data,
    resample_and_reshape,
    to_network_volume,
)

LOG = logging.getLogger(__name__)

# constants
DIAGNOSIS_MAP = {"ACV": 1}
DIAGNOSIS_MAP_binary = {"ACV": 1}


def _read_first(group, candidates, what, name):
    """Read the first dataset found among ``candidates`` (supports 'X' and 'X/data' layouts)."""
    for key in candidates:
        if key in group:
            node = group[key]
            if isinstance(node, h5py.Dataset):
                return node[:]
            if 'data' in node:
                return node['data'][:]
    raise KeyError(f"{what} data not found for subject {name} (looked for {candidates})")


class MRI2PET_3D_Dataset(Dataset):
    """
    :param volume_size: target (H, W, D) of the volumes fed to the network (default 256x256x20).
    :param data_path: path to the .h5 file (train.h5 / valid.h5).
    :param patch_size: (H, W, D) of random 3D patches used for training, or ``None`` to use the
        whole volume. Patches are only extracted when ``training=True``.
    :param lesion_patch_prob: probability that a training patch is centered on a (random) lesion
        voxel. Lesions cover <1% of the volume, so uniformly random patches would rarely contain
        them and the lesion-weighted loss / lesion-based early stopping would starve.
    :param lesion_weight: weight of lesion voxels in the loss (background = 1).
    :param inplane_resize: resize H, W (as the 2.5D pipeline did with ``T.Resize``) instead of
        center crop / pad them.
    :param random_flip: left-right flip augmentation (applied jointly to NCCT, DWI and mask).
    """

    def __init__(
        self,
        volume_size=(256, 256, 20),
        data_path=None,
        standardized_tabular=True,
        classes=None,  # 'binary' or 'multi'
        dx_labels=('ACV',),
        ROI_mask=None,
        resample_mri=False,
        patch_size=None,
        lesion_patch_prob=0.5,
        lesion_weight=50.0,
        inplane_resize=True,
        random_flip=False,
        training=False,
    ):
        super().__init__()
        assert data_path is not None and 'h5' in data_path, 'a .h5 file is required'
        self.volume_size = tuple(int(v) for v in volume_size)
        assert len(self.volume_size) == 3, 'volume_size must be (H, W, D)'
        self.data_path = data_path
        self.standardized_tabular = standardized_tabular
        self.with_label = classes
        self.dx_labels = list(dx_labels)
        self.ROI_mask = ROI_mask
        self.resample_mri = resample_mri
        self.patch_size = tuple(int(v) for v in patch_size) if patch_size else None
        if self.patch_size is not None:
            assert all(p <= v for p, v in zip(self.patch_size, self.volume_size)), \
                f'patch_size {self.patch_size} cannot be larger than volume_size {self.volume_size}'
        self.lesion_patch_prob = float(lesion_patch_prob)
        self.lesion_weight = float(lesion_weight)
        self.inplane_resize = inplane_resize
        self.random_flip = random_flip
        self.training = training

        self._load()

    # ------------------------------------------------------------------ loading
    def _prep_ncct(self, vol):
        vol = np.nan_to_num(np.asarray(vol, dtype=np.float32))
        vol = window_ncct(vol)
        return to_network_volume(vol, self.volume_size, self.inplane_resize)

    def _prep_dwi(self, vol):
        vol = np.nan_to_num(np.asarray(vol, dtype=np.float32))
        vol = robust_scale_dwi(vol)
        return to_network_volume(vol, self.volume_size, self.inplane_resize)

    def _prep_mask(self, vol):
        vol = np.nan_to_num(np.asarray(vol, dtype=np.float32))
        return to_network_volume(vol, self.volume_size, self.inplane_resize, is_mask=True)

    def _load(self):
        mri_data, pet_data, mask_data = [], [], []
        diagnosis, tabular_data, uids = [], [], []

        self.tabular_mean, self.tabular_std = None, None

        global_roi = None
        if self.ROI_mask is not None:
            global_roi = self._prep_mask(nib.load(self.ROI_mask).get_fdata())

        print('loaded from h5 file:', self.data_path)
        with h5py.File(self.data_path, mode='r') as file:
            for name, group in tqdm(file.items(), total=len(file)):
                if name == 'stats':
                    self.tabular_mean = group['tabular/mean'][:]
                    self.tabular_std = group['tabular/stddev'][:]
                    continue

                dx = group.attrs['DX']
                if isinstance(dx, bytes):
                    dx = dx.decode()
                if dx not in self.dx_labels:
                    continue

                raw_mri = _read_first(group, ['NCCT', 'NCCT/data'], 'NCCT', name)
                raw_pet = _read_first(group, ['DWI', 'DWI/data', 'SPECT/data'], 'DWI/SPECT', name)
                try:
                    raw_mask = _read_first(group, ['MASK', 'MASK/data'], 'MASK', name)
                except KeyError:
                    raw_mask = None

                if self.resample_mri:
                    raw_mri = resample_and_reshape(raw_mri, (1.5, 1.5, 1.5), raw_pet.shape)

                if raw_mri.shape != raw_pet.shape:
                    LOG.warning('NCCT %s and DWI %s shapes differ for %s; both are resized/cropped '
                                'to volume_size independently.', raw_mri.shape, raw_pet.shape, name)

                mri = self._prep_ncct(raw_mri)
                pet = self._prep_dwi(raw_pet)
                if raw_mask is not None:
                    mask = self._prep_mask(raw_mask)
                elif global_roi is not None:
                    mask = global_roi.copy()
                else:
                    mask = np.zeros_like(pet)

                mri_data.append(mri)
                pet_data.append(pet)
                mask_data.append(mask)
                tabular_data.append(group['tabular'][:])
                diagnosis.append(dx)
                uids.append(name)

        if len(pet_data) == 0:
            raise RuntimeError(f'No subjects with DX in {self.dx_labels} found in {self.data_path}')

        self.len_data = len(pet_data)
        self._image_data_mri = mri_data
        self._image_data_pet = pet_data
        self._image_data_mask = mask_data
        self._tabular_data = tabular_data
        self._mri_uid = uids

        LOG.info('DATASET: %s', self.data_path)
        LOG.info('SAMPLES (3D volumes): %d | volume (D,H,W) = %s', self.len_data, tuple(mri_data[0].shape))
        n_les = sum(int(m.sum() > 0) for m in mask_data)
        print(f'[dataset] {self.len_data} volumes, tensor (D,H,W)={tuple(mri_data[0].shape)}, '
              f'{n_les} with lesion mask')

        self._diagnosis = [DIAGNOSIS_MAP[d] for d in diagnosis]

        # indices of lesion voxels per subject (for lesion-centered patch sampling)
        # (masks are stored as (D, H, W) arrays)
        self._lesion_voxels = [np.argwhere(m > 0.5) if m.sum() > 0 else None for m in mask_data]

    # ------------------------------------------------------------------ access
    def __len__(self):
        return self.len_data

    def _random_patch_origin(self, idx):
        """Return (d0, h0, w0) of a training patch (patch_size is given in (H, W, D))."""
        ph, pw, pd = self.patch_size
        H, W, D = self.volume_size
        lesion = self._lesion_voxels[idx]
        if lesion is not None and np.random.rand() < self.lesion_patch_prob:
            cd, ch, cw = lesion[np.random.randint(len(lesion))]
            d0 = int(np.clip(cd - pd // 2, 0, D - pd))
            h0 = int(np.clip(ch - ph // 2, 0, H - ph))
            w0 = int(np.clip(cw - pw // 2, 0, W - pw))
        else:
            d0 = np.random.randint(0, D - pd + 1)
            h0 = np.random.randint(0, H - ph + 1)
            w0 = np.random.randint(0, W - pw + 1)
        return d0, h0, w0

    def __getitem__(self, idx):
        mri = self._image_data_mri[idx]    # stored as (D, H, W)
        pet = self._image_data_pet[idx]
        mask = self._image_data_mask[idx]

        if self.training and self.patch_size is not None:
            ph, pw, pd = self.patch_size
            d0, h0, w0 = self._random_patch_origin(idx)
            sl = (slice(d0, d0 + pd), slice(h0, h0 + ph), slice(w0, w0 + pw))
            mri, pet, mask = mri[sl], pet[sl], mask[sl]

        if self.training and self.random_flip and np.random.rand() < 0.5:
            mri, pet, mask = mri[..., ::-1], pet[..., ::-1], mask[..., ::-1]

        mri = torch.from_numpy(np.ascontiguousarray(mri)).unsqueeze(0).float()
        pet = torch.from_numpy(np.ascontiguousarray(pet)).unsqueeze(0).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).float()
        # loss weights: 1 in background, ``lesion_weight`` inside the lesion
        weight = 1.0 + (self.lesion_weight - 1.0) * mask

        tabular = self._tabular_data[idx]
        if self.standardized_tabular and self.tabular_mean is not None:
            std = np.where(np.abs(self.tabular_std) < 1e-8, 1.0, self.tabular_std)  # avoid division by 0
            tabular = (tabular - self.tabular_mean) / std
        tabular = process_tabular_data(tabular)

        return {
            'mri': mri,                       # (1, D, H, W)  conditioning NCCT, [0, 1]
            'pet': pet,                       # (1, D, H, W)  target DWI,        [0, 1]
            'label': self._diagnosis[idx],
            'tabular': tabular,               # (tab_dim,)
            'weight': weight,                 # (1, D, H, W)  loss weights
            'mask': mask,                     # (1, D, H, W)  binary lesion mask
            'uid': self._mri_uid[idx],
        }
