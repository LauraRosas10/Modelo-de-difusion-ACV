"""
Trainer for the 3D NCCT -> DWI diffusion model (PASTA-style dual-arm UNet, 3D version).

Main differences with the 2.5D trainer:
  * one training sample == one 3D volume / 3D patch (B, 1, D, H, W) instead of a (B, 15|3, H, W) slice stack;
  * validation / evaluation synthesise WHOLE volumes in one diffusion run (no per-slice sampling and no
    ``reconstruct_scan_from_2_5D_slices`` averaging: inter-slice consistency is learned by the 3D convolutions);
  * metrics (PSNR / SSIM / lesion MAE / lesion SSIM) are volumetric;
  * gradient accumulation bug fixed (``zero_grad`` was called inside the accumulation loop);
  * persistent data iterator (a new DataLoader iterator -> new worker processes was created at every step);
  * optional sequential backward of the cycle-consistency terms to bound 3D activation memory.
"""
import math
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from accelerate import Accelerator
from ema_pytorch import EMA
from einops import repeat, rearrange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.datasets.dataset import MRI2PET_3D_Dataset
from src.evals.ssim import ssim3d
from src.utils.data_utils import dhw_to_hwd, to_network_volume, unscale_dwi
from src.utils.early_stopping import EarlyStopping
from src.utils.utils import build_scheduler, cycle, exists, identity, print_model_size
from src.utils.weights_2d_to_3d import inflate_state_dict_2d_to_3d

try:  # optional, only needed when calculate_fid = True
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import calculate_frechet_distance
except Exception:  # pragma: no cover
    InceptionV3 = None
    calculate_frechet_distance = None


def _unnormalize(model, x):
    """[-1, 1] -> [0, 1] (the diffusion wrapper expects [0, 1] inputs and normalises them itself)."""
    if hasattr(model, 'unnormalize'):
        return model.unnormalize(x)
    return (x + 1) / 2


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=1,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.999),
        weight_decay=1e-5,
        save_and_sample_every=1000,
        num_samples=16,  # kept for config compatibility (unused in 3D)
        results_folder='./results',
        amp=False,
        fp16=True,
        split_batches=False,
        calculate_fid=False,
        inception_block_idx=2048,
        dataset=None,
        model_cycling=False,
        tabular_cond=False,
        resume=None,
        pretrain=None,
        pretrain_inflate='center',
        test_batch_size=1,
        eval_mode=False,
        eval_h5='valid.h5',
        volume_size=(256, 256, 20),      # (H, W, D) of the volumes fed to the network
        patch_size=None,                 # (H, W, D) of training patches; None = whole volume
        lesion_patch_prob=0.5,
        lesion_weight=50.0,
        inplane_resize=True,
        random_flip=False,
        max_val_volumes=4,               # number of validation volumes synthesised at each validation
        cycle_detach=False,
        output_voxel_size=(1.5, 1.5, 1.5),
        ROI_mask=None,
        dx_labels=('ACV',),
        lr_scheduler='cosine',
        warmup_steps=0,
        min_lr_ratio=0.1,
        lr_step_size=20000,
        lr_gamma=0.5,
        lr_patience=20,
        early_stopping_patience=200,
        early_stopping_min_delta=1e-4,
        reset_patience_on_resume=True,
        num_workers=None,
        wandb_enabled=False,
    ):
        super().__init__()
        self.dataset = dataset
        self.fp16 = fp16
        self.model_cycling = model_cycling
        self.tabular_cond = tabular_cond
        self.eval_mode = eval_mode
        self.results_folder = results_folder
        self.wandb_enabled = bool(wandb_enabled)
        self.ROI_mask = ROI_mask
        self.volume_size = tuple(int(v) for v in volume_size)
        self.inplane_resize = inplane_resize
        self.max_val_volumes = max_val_volumes
        self.cycle_detach = cycle_detach
        self.output_voxel_size = tuple(output_voxel_size)

        # ---- early stopping ----
        self.early_stopping = EarlyStopping(patience=early_stopping_patience,
                                            min_delta=early_stopping_min_delta, verbose=True)
        self.reset_patience_on_resume = reset_patience_on_resume
        self.es_restored = False
        self._best_backed_up = False

        self.loader_num_workers = (0 if os.name == 'nt' else 2) if num_workers is None else num_workers
        self.loader_pin_memory = False if os.name == 'nt' else torch.cuda.is_available()

        if not eval_mode:
            os.makedirs(results_folder, exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(results_folder, 'logs'))

        # ---- accelerator (fp16 mixed precision needs a GPU) ----
        use_fp16 = bool(fp16) and torch.cuda.is_available()
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if use_fp16 else 'no',
        )
        self.accelerator.native_amp = amp
        self.use_autocast = use_fp16

        # ---- model ----
        self.model = diffusion_model
        self.channels = diffusion_model.model.out_channels

        if pretrain:
            ignored_weights = ['input_blocks.0.0.weight', 'out.2.weight', 'out.2.bias']
            self.load_pretrained_model(pretrain, ignored_weights, inflate_mode=pretrain_inflate)
            print('===load pretrained model successfully===')

        # ---- InceptionV3 for (slice-wise) FID ----
        self.inception_v3 = None
        if calculate_fid:
            assert InceptionV3 is not None, 'pytorch-fid is required for calculate_fid'
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # ---- data ----
        if dataset != 'MRI2PET':
            raise NotImplementedError(f'unknown dataset {dataset}')

        common = dict(volume_size=self.volume_size, dx_labels=dx_labels, ROI_mask=ROI_mask,
                      lesion_weight=lesion_weight, inplane_resize=inplane_resize)
        if not eval_mode:
            self.ds = MRI2PET_3D_Dataset(
                data_path=os.path.join(folder, 'train.h5'), patch_size=patch_size,
                lesion_patch_prob=lesion_patch_prob, random_flip=random_flip, training=True, **common)
            # validation always uses the WHOLE volume (that is what is synthesised at test time)
            self.ds_valid = MRI2PET_3D_Dataset(
                data_path=os.path.join(folder, 'valid.h5'), patch_size=None, training=False, **common)
            self.dl_valid = DataLoader(
                self.ds_valid, batch_size=1, shuffle=False,
                pin_memory=self.loader_pin_memory, num_workers=0, drop_last=False)
        else:
            ds_test = MRI2PET_3D_Dataset(
                data_path=os.path.join(folder, eval_h5), patch_size=None, training=False, **common)
            self.dl_test = DataLoader(
                ds_test, batch_size=test_batch_size, shuffle=False,
                pin_memory=self.loader_pin_memory, num_workers=0, drop_last=False)

        # ---- EMA ----
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        if not eval_mode:
            dl = DataLoader(
                self.ds, batch_size=train_batch_size, shuffle=True,
                pin_memory=self.loader_pin_memory, num_workers=self.loader_num_workers,
                drop_last=False, persistent_workers=self.loader_num_workers > 0)
            self.dl = self.accelerator.prepare(dl)
            self.dl_iter = cycle(self.dl)

            self.opt = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr,
                                         betas=adam_betas, weight_decay=weight_decay)
            self.lr_scheduler_type = lr_scheduler
            self.scheduler = build_scheduler(
                self.opt, lr_scheduler, train_num_steps,
                warmup_steps=warmup_steps, min_lr_ratio=min_lr_ratio,
                step_size=lr_step_size, gamma=lr_gamma, patience=lr_patience)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)
            self.step = 0

            if resume:
                self.load(self.step, path=resume)
                print('resuming from model successfully')

            self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        print_model_size(self.model)

    # ------------------------------------------------------------------ utils
    @property
    def device(self):
        return self.accelerator.device

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.use_autocast)

    def save(self, milestone, model_name='model.pt'):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'early_stopping': self.early_stopping.state_dict(),
        }
        # atomic write: never leaves a half-written .pt if the process is killed
        tmp_path = str(self.results_folder / (model_name + '.tmp'))
        torch.save(data, tmp_path)
        os.replace(tmp_path, str(self.results_folder / model_name))

    def load(self, milestone=0, path=None):
        accelerator = self.accelerator
        device = accelerator.device
        if path is not None:
            data = torch.load(path, map_location=device, weights_only=False)
        else:
            data = torch.load(str(self.results_folder / 'model.pt'), map_location=device, weights_only=False)

        model = self.accelerator.unwrap_model(self.model)
        msg = model.load_state_dict(data['model'], strict=False)
        print('======load checkpoint successfully========')
        print('missing keys:', msg.missing_keys)
        print('Unexpected keys:', msg.unexpected_keys)

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if self.scheduler is not None and data.get('scheduler') is not None:
            self.scheduler.load_state_dict(data['scheduler'])

        es_state = data.get('early_stopping')
        if es_state is not None:
            self.early_stopping.load_state_dict(es_state, reset_patience=self.reset_patience_on_resume)
            self.es_restored = True
            print(f'[EarlyStopping] restored: best lesion_mae={self.early_stopping.best_lesion_mae:.6f} '
                  f'at step {self.early_stopping.best_step}')
        else:
            print('[WARN] checkpoint has no early-stopping state (old checkpoint).')

        if self.accelerator.is_main_process:
            msg_ema = self.ema.load_state_dict(data['ema'], strict=False)
            print('ema missing keys:', msg_ema.missing_keys)
            print('ema Unexpected keys:', msg_ema.unexpected_keys)

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        print('loaded model successfully from step:', self.step)

    def load_pretrained_model(self, path, ignore, inflate_mode='center'):
        """
        Load pretrained weights into the 3D model. Accepts
          * a raw UNet state dict (keys like ``input_blocks.0.0.weight``; loaded into ``diffusion.model``), or
          * a full checkpoint written by ``save`` (``{'model': {...}}``).
        2D conv kernels are inflated to 3D (see ``weights_2d_to_3d``); tensors whose shape cannot be
        reconciled (first/last conv when the number of channels changed) are skipped and reported.
        """
        pretrained = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(pretrained, dict) and 'model' in pretrained and isinstance(pretrained['model'], dict):
            sd, prefix = pretrained['model'], ''
        else:
            sd, prefix = pretrained, 'model.'

        ignore_full = {prefix + k for k in ignore}
        sd = {k: v for k, v in sd.items() if (prefix + k if prefix else k) not in ignore_full}

        model = self.accelerator.unwrap_model(self.model)
        new_sd, report = inflate_state_dict_2d_to_3d(sd, model.state_dict(), mode=inflate_mode, prefix=prefix)
        msg = model.load_state_dict(new_sd, strict=False)
        print(f"pretrained: copied={report['copied']} inflated_2d_to_3d={report['inflated']} "
              f"skipped={len(report['skipped'])}")
        for k, why in report['skipped'][:20]:
            print(f'   skipped {k}: {why}')
        print('missing keys:', len(msg.missing_keys), '| unexpected keys:', len(msg.unexpected_keys))

    # ------------------------------------------------------------------ metrics
    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)
        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...').cpu().numpy()
        return np.mean(features, axis=0), np.cov(features, rowvar=False)

    def fid_score(self, real_samples, fake_samples):
        """slice-wise FID: volumes (N, 1, D, H, W) are flattened to (N*D, 1, H, W)."""
        def flat(t):
            t = rearrange(t, 'n c d h w -> (n d) c h w')
            return repeat(t, 'b 1 h w -> b c h w', c=3) if t.shape[1] == 1 else t
        real_samples, fake_samples = flat(real_samples), flat(fake_samples)
        n = min(real_samples.shape[0], fake_samples.shape[0])
        m1, s1 = self.calculate_activation_statistics(real_samples[:n])
        m2, s2 = self.calculate_activation_statistics(fake_samples[:n])
        return calculate_frechet_distance(m1, s1, m2, s2)

    @staticmethod
    def psnr(img1, img2):
        """Volumetric PSNR per sample for tensors (N, C, D, H, W) in [0, 1]."""
        n = min(img1.shape[0], img2.shape[0])
        img1, img2 = img1[:n], img2[:n]
        mse = torch.mean((img1 - img2) ** 2, dim=list(range(1, img1.dim())))
        return 10 * torch.log10(1.0 / mse.clamp(min=1e-10))

    @staticmethod
    def volume_metrics(pred, gt, mask):
        """pred, gt, mask: (1, 1, D, H, W) on the same device, intensities in [0, 1]."""
        err = (pred - gt).abs()
        out = {
            'mae': float(err.mean()),
            'mse': float(((pred - gt) ** 2).mean()),
            'psnr': float(Trainer.psnr(gt, pred)[0]),
            'ssim': float(ssim3d(gt, pred)),
            'lesion_px': int((mask > 0.5).sum()),
        }
        lesion = (mask > 0.5).float()
        if out['lesion_px'] > 0:
            out['lesion_abs_sum'] = float((err * lesion).sum())
            out['lesion_mae'] = out['lesion_abs_sum'] / out['lesion_px']
            out['lesion_ssim'] = float(ssim3d(gt, pred, mask=lesion))
        return out

    # ------------------------------------------------------------------ preview
    @staticmethod
    def save_preview(path, mri, gt, pred, mask):
        """Save NCCT | GT DWI | synthetic DWI | |error| for the axial slice with most lesion (or the central one).
        All tensors are (1, 1, D, H, W) in [0, 1]."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:  # pragma: no cover
            return
        mri, gt, pred, mask = [t[0, 0].detach().float().cpu().numpy() for t in (mri, gt, pred, mask)]
        areas = mask.reshape(mask.shape[0], -1).sum(1)
        k = int(areas.argmax()) if areas.max() > 0 else mask.shape[0] // 2
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
        for a, img, title, kw in zip(ax, [mri[k], gt[k], pred[k], np.abs(pred[k] - gt[k])],
                                     [f'NCCT (slice {k})', 'DWI real', 'DWI sintetica', '|error|'],
                                     [dict(cmap='gray', vmin=0, vmax=1)] * 3 + [dict(cmap='magma', vmin=0, vmax=0.5)]):
            a.imshow(img, **kw)
            a.set_title(title, fontsize=9)
            a.axis('off')
        if mask[k].any():
            ax[2].contour(mask[k], levels=[0.5], colors='cyan', linewidths=0.6)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------ train
    def _cycle_losses(self, mri, pet, tab, weight, out_main):
        """
        CycleEx terms (NCCT <-> DWI), same four passes as the 2.5D trainer.
        Returns a list of scalar losses (to be summed, or back-propagated one by one when cycle_detach).
        """
        detach = self.cycle_detach
        maybe = (lambda t: t.detach()) if detach else (lambda t: t)

        syn_pet = maybe(_unnormalize(self.model, out_main['model_output']))
        out_c = self.model(mri, cond=syn_pet, tab_cond=tab, model_cycle=True, loss_weight_mask=weight)
        loss_cycle = out_c['loss'].mean()
        self.writer.add_scalar('train/train_loss_cycle', loss_cycle.item(), self.step)

        out_r2 = self.model(mri, cond=pet, tab_cond=tab, model_cycle=True, loss_weight_mask=weight)
        loss_cycle_r2 = out_r2['loss'].mean()

        syn_mri = maybe(_unnormalize(self.model, out_r2['model_output']))
        out_c_r2 = self.model(pet, cond=syn_mri, tab_cond=tab, model_cycle=False, loss_weight_mask=weight)
        loss_c_r2 = out_c_r2['loss'].mean()
        return [loss_cycle, loss_cycle_r2, loss_c_r2]

    def train(self):
        accelerator = self.accelerator
        stop = False

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps and not stop:
                total_loss = 0.
                loss_dif_val = 0.
                # BUGFIX: zero the gradients ONCE per optimiser step (the old code zeroed them inside the
                # accumulation loop, which silently disabled gradient accumulation).
                self.opt.zero_grad(set_to_none=True)

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl_iter)
                    mri, pet, weight = batch['mri'], batch['pet'], batch['weight']
                    tab = batch['tabular'] if self.tabular_cond else None

                    with self._autocast():
                        out_main = self.model(pet, cond=mri, tab_cond=tab, loss_weight_mask=weight)
                        loss_dif = out_main['loss'].mean()
                        loss_dif_val = loss_dif.item()
                        self.writer.add_scalar('train/train_loss_dif', loss_dif_val, self.step)

                        if self.model_cycling and self.cycle_detach:
                            # memory-friendly: backward every term as soon as it is computed so that each
                            # autograd graph is freed before the next 3D forward.
                            accelerator.backward(loss_dif / self.gradient_accumulate_every)
                            total_loss += loss_dif_val / self.gradient_accumulate_every
                            for term in self._cycle_losses(mri, pet, tab, weight, out_main):
                                accelerator.backward(term / self.gradient_accumulate_every)
                                total_loss += term.item() / self.gradient_accumulate_every
                            continue

                        loss = loss_dif
                        if self.model_cycling:
                            loss = loss + sum(self._cycle_losses(mri, pet, tab, weight, out_main))
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                pbar.set_description(f'loss: {loss_dif_val:.6f}')
                self.writer.add_scalar('train/train_loss', total_loss, self.step)
                accelerator.wait_for_everyone()
                self.opt.step()
                if self.scheduler is not None and self.lr_scheduler_type != 'plateau':
                    self.scheduler.step()
                self.writer.add_scalar('train/lr', self.opt.param_groups[0]['lr'], self.step)
                if self.wandb_enabled and accelerator.is_main_process:
                    import wandb
                    wandb.log({
                        'global_step': self.step,
                        'train/loss_total': total_loss,
                        'train/loss_diffusion': loss_dif_val,
                        'train/lr': self.opt.param_groups[0]['lr'],
                        'system/gpu_memory_allocated_gb': (torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0),
                        'system/gpu_memory_reserved_gb': (torch.cuda.memory_reserved() / (1024 ** 3) if torch.cuda.is_available() else 0.0),
                    })
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        stop = self.validate()

                pbar.update(1)

        accelerator.print('training complete')

    @torch.no_grad()
    def validate(self):
        """Synthesise (EMA model) ``max_val_volumes`` whole validation volumes and compute volumetric metrics.
        Returns True if early stopping asks to stop."""
        device = self.device
        accelerator = self.accelerator
        self.ema.ema_model.eval()
        milestone = self.step // self.save_and_sample_every

        val_losses, per_vol, preview = [], [], None
        real_list, fake_list = [], []

        for i, batch in enumerate(self.dl_valid):
            if i >= self.max_val_volumes:
                break
            mri, pet = batch['mri'].to(device), batch['pet'].to(device)
            weight, mask = batch['weight'].to(device), batch['mask'].to(device)
            tab = batch['tabular'].to(device) if self.tabular_cond else None

            loss = self.ema.ema_model(pet, cond=mri, tab_cond=tab, loss_weight_mask=weight)['loss'].mean()
            val_losses.append(float(loss))

            sample = self.ema.ema_model.sample(shape=pet.shape, cond=mri, tab_cond=tab, progress=False)
            m = self.volume_metrics(sample, pet, mask)
            per_vol.append(m)
            if self.inception_v3 is not None:
                real_list.append(pet.cpu())
                fake_list.append(sample.cpu())
            if preview is None or (m['lesion_px'] > 0 and preview[-1] == 0):
                preview = (mri.cpu(), pet.cpu(), sample.cpu(), mask.cpu(), m['lesion_px'])
            del sample
            torch.cuda.empty_cache()

        valid_loss = float(np.mean(val_losses))
        print(f'valid loss: {valid_loss:.6f}')
        self.writer.add_scalar('val/val_loss', valid_loss, self.step)
        if self.wandb_enabled and accelerator.is_main_process:
            import wandb
            wandb.log({'global_step': self.step, 'val/diffusion_loss': valid_loss})

        mean = lambda key: float(np.mean([m[key] for m in per_vol]))
        recon_l1, recon_l2 = mean('mae'), mean('mse')
        psnr_score, ssim_score = mean('psnr'), mean('ssim')
        print(f'valid recon L2 loss: {recon_l2:.5f}')
        print(f'valid recon L1 loss: {recon_l1:.5f}')

        if self.scheduler is not None and self.lr_scheduler_type == 'plateau':
            self.scheduler.step(recon_l1)

        scores = str(self.results_folder / 'scores.txt')
        with open(scores, 'a') as f:
            f.write(f'\nmilestone: {milestone}')
            f.write(f'\nvalid recon L1 loss: {recon_l1:.5f}')
            f.write(f'\nvalid recon L2 loss: {recon_l2:.5f}')

        self.writer.add_scalar('val/L1', recon_l1, self.step)
        self.writer.add_scalar('val/psnr_3d', psnr_score, self.step)
        self.writer.add_scalar('val/ssim_3d', ssim_score, self.step)
        if self.wandb_enabled and accelerator.is_main_process:
            import wandb
            wandb.log({
                'val/reconstruction_mae': recon_l1,
                'val/reconstruction_mse': recon_l2,
                'val/psnr_3d': psnr_score,
                'val/ssim_3d': ssim_score,
                'global_step': self.step,
            })

        # (optional) preview image
        if preview is not None:
            self.save_preview(str(self.results_folder / f'preview-{milestone}.png'), *preview[:4])

        # checkpoint of the last state
        self.save(milestone)

        if self.inception_v3 is not None:
            fid = self.fid_score(torch.cat(real_list).to(device), torch.cat(fake_list).to(device))
            accelerator.print(f'fid_score (slice-wise): {fid}')
            with open(scores, 'a') as f:
                f.write(f'\nfid_score: {fid}')

        accelerator.print(f'psnr_score (3D): {psnr_score}')
        accelerator.print(f'ssim_score (3D): {ssim_score}')
        with open(scores, 'a') as f:
            f.write(f'\npsnr_score: {psnr_score}')
            f.write(f'\nssim_score: {ssim_score}\n')

        # ----------------------------- lesion metrics / early stopping -----------------------------
        n_les = sum(m['lesion_px'] for m in per_vol)
        if n_les > 0:
            with_les = [m for m in per_vol if m['lesion_px'] > 0]
            lesion_mae = sum(m['lesion_abs_sum'] for m in with_les) / n_les           # pooled over voxels
            lesion_ssim = sum(m['lesion_ssim'] * m['lesion_px'] for m in with_les) / n_les
            accelerator.print(f'lesion_mae: {lesion_mae:.6f} | lesion_ssim: {lesion_ssim:.6f} | lesion px: {n_les}')
            self.writer.add_scalar('val/lesion_mae', lesion_mae, self.step)
            self.writer.add_scalar('val/lesion_ssim', lesion_ssim, self.step)
            if self.wandb_enabled and accelerator.is_main_process:
                import wandb
                wandb.log({'global_step': self.step, 'val/lesion_mae': lesion_mae, 'val/lesion_ssim': lesion_ssim, 'val/lesion_voxels': n_les})
            with open(scores, 'a') as f:
                f.write(f'\nlesion_mae: {lesion_mae:.6f}\nlesion_ssim: {lesion_ssim:.6f}\nlesion_px: {n_les}\n')

            should_stop = self.early_stopping(lesion_mae=lesion_mae, ssim=ssim_score, mae=recon_l1, step=self.step)

            if self.early_stopping.counter == 0:  # improved
                best_path = self.results_folder / 'best_val_model.pt'
                if (not self.es_restored) and best_path.exists() and not self._best_backed_up:
                    shutil.copy(best_path, self.results_folder / 'best_val_model_prev.pt')
                    self._best_backed_up = True
                self.save(milestone, model_name='best_val_model.pt')
                with open(scores, 'a') as f:
                    f.write(f'\n************Best val model (lesion_mae), milestone: {milestone}*************\n')

            if should_stop:
                accelerator.print('Early stopping activated.')
                self.save(milestone)
                return True
        else:
            accelerator.print(f'[INFO] Milestone {milestone}: no lesion voxels in the validation volumes used. '
                              f'best_val_model.pt not updated.')
            with open(scores, 'a') as f:
                f.write('\nlesion_mae: None\nlesion_ssim: None\nlesion_px: 0\n')
        return False

    # ------------------------------------------------------------------ evaluate
    @torch.no_grad()
    def evaluate(self, checkpoint, evaluate_folder, synthesis=False, synthesis_folder=None, get_ROI_loss=False):
        self.evaluate_folder = Path(evaluate_folder)
        self.evaluate_folder.mkdir(exist_ok=True, parents=True)

        # BUGFIX: ``already_done`` was only defined in synthesis mode but used in both modes
        already_done = set()
        if synthesis:
            assert synthesis_folder is not None, 'synthesis_folder is None'
            syn_folder = Path(synthesis_folder)
            syn_folder.mkdir(exist_ok=True, parents=True)
            scores_file = str(syn_folder / 'all_eval_scores.txt')
            if os.path.exists(scores_file):
                with open(scores_file, 'r') as f:
                    for line in f:
                        if 'NCCT -> DWI:' in line:
                            already_done.add(line.split(':')[1].strip().split('_')[0])
                print(f'Patients already processed: {len(already_done)} -> {already_done}')

        roi_global = None
        if get_ROI_loss:
            assert self.ROI_mask is not None, 'ROI_mask is None'
            roi_global = torch.from_numpy(
                to_network_volume(nib.load(self.ROI_mask).get_fdata(), self.volume_size,
                                  self.inplane_resize, is_mask=True)).to(self.device)[None, None]
            print('use ROI loss, ROI_mask_shape:', tuple(roi_global.shape))

        data = torch.load(checkpoint, map_location=self.device, weights_only=False)
        msg = self.model.load_state_dict(data['model'], strict=False)
        print('======load checkpoint successfully========')
        print('missing keys:', msg.missing_keys)
        print('Unexpected keys:', msg.unexpected_keys)
        self.step = data['step']
        msg_ema = self.ema.load_state_dict(data['ema'], strict=False)
        print('ema missing keys:', msg_ema.missing_keys)
        print('ema Unexpected keys:', msg_ema.unexpected_keys)
        print('loaded model successfully from:', checkpoint)

        self.model.eval()
        self.ema.ema_model.eval()
        ema = self.ema.ema_model

        out_dir = Path(synthesis_folder) if synthesis else self.evaluate_folder
        scores_path = str(out_dir / ('all_eval_scores.txt' if synthesis else 'eval_scores.txt'))
        affine = np.diag(list(self.output_voxel_size) + [1.0])

        all_metrics, roi_losses = [], []

        for batch in self.dl_test:
            uids = [str(u) for u in batch['uid']]
            if all(u in already_done for u in uids):
                print(f'Already exists, skipping: {uids}')
                continue

            mri, pet = batch['mri'].to(self.device), batch['pet'].to(self.device)
            mask = batch['mask'].to(self.device)
            tab = batch['tabular'].to(self.device) if self.tabular_cond else None

            # one diffusion run synthesises the WHOLE volume (B, 1, D, H, W)
            sample = ema.sample(shape=pet.shape, cond=mri, tab_cond=tab, progress=True)

            cycle_out = None
            if self.model_cycling and not synthesis:
                cyc = ema(mri, cond=sample, tab_cond=tab, model_cycle=True)
                cycle_out = _unnormalize(ema, cyc['model_output']).clamp(0, 1)

            for b, uid in enumerate(uids):
                if uid in already_done:
                    continue
                label = 'ACV'
                s_b, p_b, m_b, k_b = sample[b:b + 1], pet[b:b + 1], mri[b:b + 1], mask[b:b + 1]
                met = self.volume_metrics(s_b, p_b, k_b)
                met['uid'] = uid
                all_metrics.append(met)

                syn_unscaled = unscale_dwi(dhw_to_hwd(s_b[0, 0].cpu().numpy()))
                nib.Nifti1Image(syn_unscaled, affine=affine).to_filename(
                    str(out_dir / f'{uid}_{label}_syn_dwi.nii.gz' if synthesis
                        else out_dir / f'whole_sample_dwi_{uid}_{label}.nii.gz'))

                if not synthesis:
                    gt_unscaled = unscale_dwi(dhw_to_hwd(p_b[0, 0].cpu().numpy()))
                    nib.Nifti1Image(gt_unscaled, affine=affine).to_filename(
                        str(out_dir / f'whole_GT_dwi_{uid}_{label}.nii.gz'))
                    nib.Nifti1Image(syn_unscaled - gt_unscaled, affine=affine).to_filename(
                        str(out_dir / f'error_map_{uid}_{label}.nii.gz'))
                    self.save_preview(str(out_dir / f'preview_{uid}.png'), m_b, p_b, s_b, k_b)

                lines = [f'NCCT -> DWI: {uid}_{label}: {met["mae"]:.6f}',
                         f'NCCT -> DWI PSNR/SSIM(3D): {uid}_{label}: {met["psnr"]:.4f} / {met["ssim"]:.4f}']
                if 'lesion_mae' in met:
                    lines.append(f'NCCT -> DWI lesion MAE/SSIM: {uid}_{label}: '
                                 f'{met["lesion_mae"]:.6f} / {met["lesion_ssim"]:.4f} ({met["lesion_px"]} vox)')
                if roi_global is not None:
                    roi_loss = float(((s_b - p_b).abs() * roi_global).mean())
                    roi_losses.append(roi_loss)
                    lines.append(f'NCCT -> DWI ROI loss: {uid}_{label}: {roi_loss:.6f}')
                if cycle_out is not None:
                    l1_mri = float((cycle_out[b:b + 1] - m_b).abs().mean())
                    lines.append(f'DWI -> NCCT: {uid}_{label}: {l1_mri:.6f}')
                for ln in lines:
                    print(ln)
                with open(scores_path, 'a') as f:
                    f.write('\n' + '\n'.join(lines))

            torch.cuda.empty_cache()

        if all_metrics:
            agg = {k: float(np.mean([m[k] for m in all_metrics])) for k in ('mae', 'mse', 'psnr', 'ssim')}
            with_les = [m for m in all_metrics if 'lesion_mae' in m]
            summary = [f'test recon L1 loss: {agg["mae"]:.6f}', f'test recon L2 loss: {agg["mse"]:.6f}',
                       f'psnr_score (3D): {agg["psnr"]:.4f}', f'ssim_score (3D): {agg["ssim"]:.4f}']
            if with_les:
                n = sum(m['lesion_px'] for m in with_les)
                summary.append(f'lesion_mae (pooled): {sum(m["lesion_abs_sum"] for m in with_les) / n:.6f}')
                summary.append(f'lesion_ssim (voxel-weighted): '
                               f'{sum(m["lesion_ssim"] * m["lesion_px"] for m in with_les) / n:.4f}')
            if roi_losses:
                summary.append(f'test ROI loss: {sum(roi_losses) / len(roi_losses):.6f}')
            print('\n'.join(summary))
            with open(str(self.evaluate_folder / 'eval_scores.txt'), 'a') as f:
                f.write('\n=== SUMMARY ===\n' + '\n'.join(summary) + '\n')
