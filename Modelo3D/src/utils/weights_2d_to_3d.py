"""
Initialise a 3D UNet from the weights of a 2D / 2.5D UNet ("inflation", Carreira & Zisserman, CVPR 2017).

A Conv2d kernel (O, I, kh, kw) becomes a Conv3d kernel (O, I, kd, kh, kw):

* ``mode='center'`` : the 2D kernel goes in the central depth plane, the other planes are 0.
  The 3D network initially behaves exactly like the 2D network applied slice by slice (no
  mixing between slices) and learns the inter-slice interaction during training.
* ``mode='mean'``   : the 2D kernel is replicated along depth and divided by kd (the response to a
  depth-constant input equals the 2D response).

Tensors whose shapes already match (norms, linears, 1x1 convs in attention, embeddings...) are copied;
tensors whose shape cannot be reconciled (e.g. the first / last conv when the number of input /
output channels changed, as when going from 15 slice-channels to 1 volume-channel) are skipped.
"""
import torch


def inflate_state_dict_2d_to_3d(sd_2d, sd_3d, mode='center', prefix=''):
    """
    :param sd_2d: pretrained 2D state dict (keys without ``prefix``).
    :param sd_3d: state dict of the target 3D module (``module.state_dict()``); keys are prefixed with ``prefix``.
    :return: (new_state_dict, report) where new_state_dict only holds loadable tensors.
    """
    assert mode in ('center', 'mean'), mode
    out, copied, inflated, skipped = {}, [], [], []
    for k, v in sd_2d.items():
        k3 = prefix + k
        if k3 not in sd_3d:
            skipped.append((k, 'missing in 3D model'))
            continue
        t3 = sd_3d[k3]
        if v.shape == t3.shape:
            out[k3] = v
            copied.append(k)
        elif v.dim() == 4 and t3.dim() == 5 and v.shape[:2] == t3.shape[:2] and v.shape[2:] == t3.shape[3:]:
            kd = t3.shape[2]
            if mode == 'center':
                w = torch.zeros_like(t3)
                w[:, :, kd // 2] = v
            else:
                w = v.unsqueeze(2).repeat(1, 1, kd, 1, 1) / kd
            out[k3] = w
            inflated.append(k)
        else:
            skipped.append((k, f'shape {tuple(v.shape)} vs {tuple(t3.shape)}'))
    report = {'copied': len(copied), 'inflated': len(inflated), 'skipped': skipped}
    return out, report
