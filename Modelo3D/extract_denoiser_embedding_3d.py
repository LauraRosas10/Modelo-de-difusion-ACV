#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extrae embeddings del denoiser 3D de PASTA para NCCT -> DWI (ACV).

Salidas:
  patients/<uid>.npz:
    local             (G, C_mid) tokens 3D del cuello de botella, orden (D,H,W)
    slice_vec         (D, C_mid) un vector por posición axial del mapa mid
    slice_idx         (D,) índices aproximados en la profundidad de entrada
    global_mean       (Ds,) GAP de las capas seleccionadas (y GMP si --pool meanmax)
    global_max        (Ds,) máximo por canal sobre los vectores por corte
    label, n_total_slices, input_shape
  patients.csv, embeddings_patient.npy, meta.json

IMPORTANTE:
- El dataset 3D devuelve un volumen completo por paciente: (1,D,H,W).
- --source noise es la opción recomendada si las representaciones no deben depender del DWI real.
- --source gt usa q_sample(DWI,t) y por tanto introduce información del objetivo real.
- Los índices de corte se aproximan al mapa de cuello de botella si la arquitectura modifica D.


python extract_denoiser_embedding_3d.py \
    --config src/config/pasta3d_acv_eval.yaml \
    --split test \
    --source noise \
    --t last \
    --layers mid \
    --pool meanmax \
    --grid 4 6 6 \
    --out results_acv_3d/embeddings_test
"""
import argparse
import csv
import json
import os
import re

import h5py  # requerido indirectamente por el dataset
import numpy as np
import torch
import torch.nn.functional as F
from ema_pytorch import EMA
from torch.utils.data import DataLoader

from src.diffusion.gaussian_diffusion import ModelMeanType, ModelVarType, LossType
from src.diffusion.respace import SpacedDiffusion, space_timesteps
from src.model.unet import UNetModel
from src.datasets.dataset import MRI2PET_3D_Dataset
from src.utils.utils import load_config_from_yaml

OBJECTIVE = {
    'PREVIOUS_X': ModelMeanType.PREVIOUS_X,
    'START_X': ModelMeanType.START_X,
    'EPSILON': ModelMeanType.EPSILON,
    'VELOCITY': ModelMeanType.VELOCITY,
}
MODEL_VAR_TYPE = {
    'LEARNED': ModelVarType.LEARNED,
    'FIXED_SMALL': ModelVarType.FIXED_SMALL,
    'FIXED_LARGE': ModelVarType.FIXED_LARGE,
    'LEARNED_RANGE': ModelVarType.LEARNED_RANGE,
}
LOSS_TYPE = {'l1': LossType.MAE, 'l2': LossType.MSE}


def build_diffusion(args):
    """Construye exactamente la U-Net 3D y la difusión de train_mri2pet.py."""
    volume_size = list(args.volume_size)  # YAML: (H,W,D)
    image_size = [volume_size[2], volume_size[0], volume_size[1]]  # red: (D,H,W)
    use_checkpoint = getattr(args, 'use_checkpoint', False)
    attention_type_3d = getattr(args, 'attention_type_3d', 'project_excite')

    model = UNetModel(
        image_size=image_size,
        in_channels=args.model_in_channels,
        model_channels=args.unet_dim,
        out_channels=args.out_channels_model,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        channel_mult=args.unet_dim_mults,
        resblock_updown=args.resblock_updown,
        dims=args.dims,
        dropout=args.dropout,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_condition=True,
        use_time_condition=args.use_time_condition,
        cond_emb_channels=args.cond_emb_channels,
        tab_cond_dim=args.tab_cond_dim,
        use_tabular_cond=args.use_tabular_cond_model,
        with_attention=args.with_attention,
        cond_apply_method=args.cond_apply_method,
        use_checkpoint=use_checkpoint,
        attention_type_3d=attention_type_3d,
    )
    encoder = UNetModel(
        image_size=image_size,
        in_channels=args.encoder_in_channels,
        model_channels=args.unet_dim,
        out_channels=args.out_channels_encoder,
        num_res_blocks=args.num_res_blocks,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        channel_mult=args.unet_dim_mults,
        resblock_updown=args.resblock_updown,
        dims=args.dims,
        dropout=args.dropout,
        use_fp16=False,
        use_scale_shift_norm=True,
        use_condition=True,
        use_time_condition=args.use_time_condition,
        tab_cond_dim=args.tab_cond_dim,
        use_tabular_cond=args.use_tabular_cond_encoder,
        with_attention=args.with_attention,
        cond_apply_method=args.cond_apply_method,
        use_checkpoint=use_checkpoint,
        attention_type_3d=attention_type_3d,
    )
    if args.dims != 3:
        raise ValueError(f'Este extractor requiere dims: 3; configuración actual: {args.dims}')

    return SpacedDiffusion(
        use_timesteps=space_timesteps(args.timesteps, args.timestep_respacing),
        model=model,
        encoder=encoder,
        beta_schedule=args.beta_schedule,
        timesteps=args.timesteps,
        model_mean_type=OBJECTIVE[args.objective],
        model_var_type=MODEL_VAR_TYPE[args.model_var_type],
        loss_type=LOSS_TYPE[args.loss_type],
        gen_type=args.gen_type,
        use_fp16=False,
        condition=args.condition,
        reconstructed_loss=args.reconstructed_loss,
        recon_weight=args.recon_weight,
        rescale_intensity=args.rescale_intensity,
    )


def load_weights(diffusion, checkpoint, weights, device, ema_decay):
    ckpt = torch.load(checkpoint, map_location=device)
    if weights == 'ema':
        if 'ema' not in ckpt:
            raise KeyError("El checkpoint no contiene la clave 'ema'; pruebe --weights model.")
        ema = EMA(diffusion, beta=ema_decay, update_every=10)
        msg = ema.load_state_dict(ckpt['ema'], strict=False)
        print(f'EMA missing keys ({len(msg.missing_keys)}):', msg.missing_keys[:20])
        print(f'EMA unexpected keys ({len(msg.unexpected_keys)}):', msg.unexpected_keys[:20])
        net = ema.ema_model
    else:
        if 'model' not in ckpt:
            raise KeyError("El checkpoint no contiene la clave 'model'.")
        msg = diffusion.load_state_dict(ckpt['model'], strict=False)
        print(f'model missing keys ({len(msg.missing_keys)}):', msg.missing_keys[:20])
        print(f'model unexpected keys ({len(msg.unexpected_keys)}):', msg.unexpected_keys[:20])
        net = diffusion
    net.to(device).eval()
    return net


def gap(feat):
    """GAP para mapas 2D o 3D: (B,C,...) -> (B,C)."""
    return feat.mean(dim=tuple(range(2, feat.dim())))


def gmp(feat):
    """GMP para mapas 2D o 3D: (B,C,...) -> (B,C)."""
    return feat.amax(dim=tuple(range(2, feat.dim())))


def pick_layers(features, denoiser, layers, pool):
    """Concatenación de GAP (y opcionalmente GMP) de capas elegidas."""
    mid_idx = len(denoiser.input_blocks) + 1
    if layers == 'mid':
        chosen = [features[mid_idx]]
    elif layers == 'mid_dec':
        chosen = [features[mid_idx]] + features[mid_idx + 1:]
    elif layers == 'all':
        chosen = features[1:]
    else:
        raise ValueError(f'Valor --layers desconocido: {layers}')
    pooled = [gap(f) for f in chosen]
    if pool == 'meanmax':
        pooled += [gmp(f) for f in chosen]
    return torch.cat(pooled, dim=1)


def local_tokens_3d(feat, grid):
    """(B,C,D,H,W) -> (B,G,C), tokens ordenados por (profundidad,fila,columna)."""
    if feat.ndim != 5:
        raise ValueError(f'Se esperaba un mapa 3D (B,C,D,H,W), llegó {tuple(feat.shape)}')
    if all(int(v) > 0 for v in grid):
        feat = F.adaptive_avg_pool3d(feat, tuple(int(v) for v in grid))
    return feat.flatten(2).transpose(1, 2).contiguous()


def per_slice_embeddings(feat):
    """(B,C,D,H,W) -> (B,D,C): pooling H,W, manteniendo el eje axial D."""
    if feat.ndim != 5:
        raise ValueError(f'Se esperaba mapa 3D, llegó {tuple(feat.shape)}')
    return feat.mean(dim=(-1, -2)).transpose(1, 2).contiguous()


@torch.no_grad()
def embed_volume(diffusion, mri, pet, tab, t_list, source, fixed_noise, layers, pool, grid):
    """Extrae embeddings para un volumen. mri/pet: (B,1,D,H,W), normalmente B=1."""
    if mri.ndim != 5:
        raise ValueError(f'NCCT debe ser (B,C,D,H,W), recibido {tuple(mri.shape)}')
    b = mri.shape[0]
    cond_in = diffusion.normalize(mri)
    _, cond = diffusion.encoder(cond_in, return_features=True, tab_cond=tab)
    denoiser = diffusion._wrap_model(diffusion.model)

    if fixed_noise is None or tuple(fixed_noise.shape[2:]) != tuple(mri.shape[2:]):
        raise ValueError(f'Ruido fijo {None if fixed_noise is None else tuple(fixed_noise.shape)} '
                         f'no coincide con volumen {tuple(mri.shape)}')
    noise = fixed_noise.expand(b, -1, -1, -1, -1).to(device=mri.device, dtype=mri.dtype)
    x0 = diffusion.normalize(pet) if source == 'gt' else None
    mid_idx = len(diffusion.model.input_blocks) + 1
    vectors, tokens, slices = [], [], []

    for t_idx in t_list:
        t = torch.full((b,), int(t_idx), device=mri.device, dtype=torch.long)
        x_t = noise if source == 'noise' else diffusion.q_sample(x0, t, noise=noise)
        _, feats = denoiser(x_t, t, cond=cond, tab_cond=tab, return_features=True)
        vectors.append(pick_layers(feats, diffusion.model, layers, pool))
        mid = feats[mid_idx]
        tokens.append(local_tokens_3d(mid, grid))
        slices.append(per_slice_embeddings(mid))

    global_vec = torch.stack(vectors, dim=0).mean(dim=0)
    local = torch.stack(tokens, dim=0).mean(dim=0)
    slice_vec = torch.stack(slices, dim=0).mean(dim=0)
    return global_vec, local, slice_vec


def parse_t_list(value, num_timesteps):
    if value == 'last':
        return [num_timesteps - 1]
    result = [int(v.strip()) for v in value.split(',') if v.strip()]
    if not result:
        raise ValueError('--t debe ser last o una lista de índices, por ejemplo 99,50,10')
    for t in result:
        if not 0 <= t < num_timesteps:
            raise ValueError(f't={t} fuera de rango [0,{num_timesteps - 1}]')
    return result


def safe_name(uid):
    cleaned = ''.join(ch if (ch.isalnum() or ch in '-_.') else '_' for ch in str(uid))
    return cleaned or 'unknown_subject'


def main():
    p = argparse.ArgumentParser(description='Extracción de embeddings PASTA 3D para NCCT -> DWI.')
    p.add_argument('--config', required=True, help='YAML 3D usado en entrenamiento/evaluación')
    p.add_argument('--checkpoint', default=None, help='Por defecto: <results_folder>/best_val_model.pt')
    p.add_argument('--weights', choices=['ema', 'model'], default='ema')
    p.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    p.add_argument('--source', choices=['noise', 'gt'], default='noise')
    p.add_argument('--t', default='last', help="Índices respaciados separados por coma o 'last'")
    p.add_argument('--layers', choices=['mid', 'mid_dec', 'all'], default='mid')
    p.add_argument('--pool', choices=['mean', 'meanmax'], default='meanmax')
    p.add_argument('--grid', type=int, nargs=3, default=[4, 6, 6], metavar=('D', 'H', 'W'),
                   help='Rejilla de tokens 3D; 0 0 0 conserva la resolución nativa')
    p.add_argument('--l2', action='store_true', help='Normaliza L2 global_mean al guardar embeddings_patient.npy')
    p.add_argument('--seed', type=int, default=666)
    p.add_argument('--out', default=None, help='Por defecto: <results_folder>/embeddings_3d')
    p.add_argument('--limit', type=int, default=0, help='Máximo de pacientes para prueba; 0 = todos')
    cli = p.parse_args()

    args = load_config_from_yaml(cli.config)
    if int(getattr(args, 'dims', 0)) != 3:
        raise ValueError(f"La configuración debe tener dims: 3; recibido {getattr(args, 'dims', None)}")
    if len(cli.grid) != 3 or any(v < 0 for v in cli.grid):
        raise ValueError('--grid requiere tres enteros no negativos; ejemplo: --grid 4 6 6')
    if any(v == 0 for v in cli.grid) and not all(v == 0 for v in cli.grid):
        raise ValueError('Use tres valores positivos o --grid 0 0 0 para resolución nativa.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = cli.checkpoint or os.path.join(args.results_folder, 'best_val_model.pt')
    out_dir = cli.out or os.path.join(args.results_folder, 'embeddings_3d')
    patient_dir = os.path.join(out_dir, 'patients')
    os.makedirs(patient_dir, exist_ok=True)
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'No existe el checkpoint: {checkpoint}')

    diffusion = build_diffusion(args)
    net = load_weights(diffusion, checkpoint, cli.weights, device, args.ema_decay)
    t_list = parse_t_list(cli.t, net.num_timesteps)
    if cli.source == 'noise' and t_list != [net.num_timesteps - 1]:
        print('AVISO: --source noise usa ruido puro; normalmente utilice --t last.')

    h5_path = os.path.join(args.data_dir, f'{cli.split}.h5')
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f'No existe el archivo del split: {h5_path}')
    ds = MRI2PET_3D_Dataset(
        volume_size=tuple(args.volume_size),
        data_path=h5_path,
        standardized_tabular=True,
        classes='binary',
        dx_labels=args.dx_labels,
        ROI_mask=getattr(args, 'ROI_mask', None),
        resample_mri=getattr(args, 'resample_mri', False),
        patch_size=None,
        lesion_patch_prob=getattr(args, 'lesion_patch_prob', 0.5),
        lesion_weight=getattr(args, 'lesion_weight', 1.0),
        inplane_resize=getattr(args, 'inplane_resize', True),
        random_flip=False,
        training=False,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Ruido determinista compartido entre sujetos; se crea con la forma de entrada de la red (D,H,W).
    vol_h, vol_w, vol_d = [int(v) for v in args.volume_size]
    shape = (1, int(args.out_channels_model), vol_d, vol_h, vol_w)
    generator = torch.Generator(device='cpu').manual_seed(cli.seed)
    fixed_noise = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)

    rows, global_vectors = [], []
    dims = {}
    print(f'Device={device}; split={cli.split}; source={cli.source}; timesteps={t_list}; grid={cli.grid}')
    print(f'H5={h5_path}; checkpoint={checkpoint}; input noise shape={shape}')

    for i, data in enumerate(dl):
        if cli.limit > 0 and i >= cli.limit:
            break
        mri = data['mri'].to(device)       # (B,1,D,H,W)
        pet = data['pet'].to(device)       # (B,1,D,H,W)
        label = int(data['label'].reshape(-1)[0].item())
        tab = data['tabular'].to(device).float() if getattr(args, 'tabular_cond', True) else None
        uid_value = data['uid'][0]
        uid = str(uid_value.item() if hasattr(uid_value, 'item') else uid_value)
        n_slices = int(mri.shape[2])

        if tab is not None and tab.ndim == 1:
            tab = tab.unsqueeze(0)
        if tab is not None and tab.shape[1] != int(args.tab_cond_dim):
            raise ValueError(f'{uid}: tabular tiene {tab.shape[1]} variables, pero tab_cond_dim={args.tab_cond_dim}')

        global_vec, tokens, slice_vec = embed_volume(
            net, mri, pet, tab, t_list, cli.source, fixed_noise,
            cli.layers, cli.pool, cli.grid,
        )
        global_vec = global_vec[0].float().cpu().numpy()
        tokens = tokens[0].float().cpu().numpy()
        slice_vec = slice_vec[0].float().cpu().numpy()

        # Asociar cada posición del eje D del cuello de botella con un índice de corte de entrada.
        # Es una correspondencia aproximada cuando la arquitectura altera la profundidad.
        if len(slice_vec) == n_slices:
            slice_idx = np.arange(n_slices, dtype=np.int16)
        else:
            slice_idx = np.rint(np.linspace(0, max(n_slices - 1, 0), len(slice_vec))).astype(np.int16)

        g_max = slice_vec.max(axis=0)
        fname = os.path.join('patients', safe_name(uid) + '.npz')
        np.savez_compressed(
            os.path.join(out_dir, fname),
            local=tokens.astype(np.float16),
            slice_vec=slice_vec.astype(np.float32),
            slice_idx=slice_idx,
            global_mean=global_vec.astype(np.float32),
            global_max=g_max.astype(np.float32),
            label=np.int64(label),
            n_total_slices=np.int64(n_slices),
            input_shape=np.asarray(mri.shape[2:], dtype=np.int16),
        )
        vec = global_vec / (np.linalg.norm(global_vec) + 1e-8) if cli.l2 else global_vec
        global_vectors.append(vec.astype(np.float32))
        rows.append([uid, label, n_slices, len(slice_vec), fname])
        dims = {
            'global_dim': int(global_vec.shape[0]),
            'slice_dim': int(slice_vec.shape[1]),
            'tokens_per_volume': int(tokens.shape[0]),
            'token_dim': int(tokens.shape[1]),
            'slice_positions_in_bottleneck': int(slice_vec.shape[0]),
        }
        print(f'{uid}: volume={tuple(mri.shape[2:])}; slice_vec={slice_vec.shape}; local={tokens.shape}')

    if not rows:
        raise RuntimeError('No se procesó ningún paciente. Revise DX, split y ruta HDF5.')
    np.save(os.path.join(out_dir, 'embeddings_patient.npy'), np.stack(global_vectors, axis=0))
    with open(os.path.join(out_dir, 'patients.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['uid', 'label', 'n_slices_total', 'n_slices_used', 'file'])
        writer.writerows(rows)
    meta = {
        'config': os.path.abspath(cli.config), 'checkpoint': os.path.abspath(checkpoint),
        'weights': cli.weights, 'split': cli.split, 'source': cli.source,
        't': t_list, 'layers': cli.layers, 'pool': cli.pool, 'grid_dhw': cli.grid,
        'seed': cli.seed, 'l2': cli.l2, 'dims': 3, 'volume_size_hwd': list(args.volume_size),
        'input_tensor_order': '(B,C,D,H,W)', 'local_token_order': '(depth,row,column)',
        'slice_vec_note': 'Pooling espacial H,W del mapa mid 3D; cada vector puede contener contexto entre cortes.',
        'slice_idx_note': 'Índices equiespaciados aproximados si la profundidad del mapa mid difiere de la entrada.',
        **dims,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'Listo: {len(rows)} pacientes guardados en {out_dir}; dimensiones={dims}')


if __name__ == '__main__':
    main()
