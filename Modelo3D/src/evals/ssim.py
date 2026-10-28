import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True, mask = None):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        # Promedio restringido a una región (p.ej. mascara de lesion), en vez de la imagen completa.
        # El mapa de SSIM sigue usando el contexto local (ventana gaussiana), solo cambia el promedio final.
        mask = mask.to(ssim_map.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (ssim_map * mask).sum() / denom

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).mean(1)
    


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True, mask = None):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average, mask = mask)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


# ---------------------------------------------------------------------------
# 3D SSIM (volumetric) -- added for the 3D architecture
# ---------------------------------------------------------------------------
def _gauss_1d(window_size, sigma=1.5, device=None, dtype=torch.float32):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _odd_window(window_size, size):
    """Largest odd window <= min(window_size, size)."""
    w = min(window_size, size)
    return w if w % 2 == 1 else max(1, w - 1)


def _separable_gauss_filter_3d(x, wd, wh, ww, groups):
    """Gaussian filtering of a [N, C, D, H, W] tensor as three 1D convolutions (same result as the
    full 3D kernel but ~window_size^2 times cheaper). Zero padding as in the 2D SSIM."""
    c = groups
    kd = wd.view(1, 1, -1, 1, 1).expand(c, 1, -1, 1, 1).contiguous()
    kh = wh.view(1, 1, 1, -1, 1).expand(c, 1, 1, -1, 1).contiguous()
    kw = ww.view(1, 1, 1, 1, -1).expand(c, 1, 1, 1, -1).contiguous()
    x = F.conv3d(x, kd, padding=(wd.numel() // 2, 0, 0), groups=c)
    x = F.conv3d(x, kh, padding=(0, wh.numel() // 2, 0), groups=c)
    x = F.conv3d(x, kw, padding=(0, 0, ww.numel() // 2), groups=c)
    return x


def ssim3d(img1, img2, window_size=11, size_average=True, mask=None):
    """
    Volumetric SSIM for tensors [N, C, D, H, W] in [0, 1].

    * Separable Gaussian window (sigma = 1.5).
    * The window along depth is clipped to the number of slices (thick-slice CT/MRI
      volumes only have ~20 slices), the in-plane window is ``window_size``.
    * Always computed in float32 with autocast disabled (variances are differences of
      large similar numbers and are unstable in fp16).
    * ``mask`` ([N, 1|C, D, H, W]) restricts the final average (e.g. to the lesion).
    """
    assert img1.dim() == 5 and img1.shape == img2.shape, (img1.shape, img2.shape)
    _, channel, d, h, w = img1.shape
    dev_type = img1.device.type
    with torch.autocast(device_type=dev_type, enabled=False):
        img1 = img1.float()
        img2 = img2.float()
        wd = _gauss_1d(_odd_window(window_size, d), device=img1.device)
        wh = _gauss_1d(_odd_window(window_size, h), device=img1.device)
        ww = _gauss_1d(_odd_window(window_size, w), device=img1.device)

        # filter the 5 required statistics in one grouped convolution
        stacked = torch.cat([img1, img2, img1 * img1, img2 * img2, img1 * img2], dim=1)
        filt = _separable_gauss_filter_3d(stacked, wd, wh, ww, groups=5 * channel)
        mu1, mu2, e11, e22, e12 = torch.split(filt, channel, dim=1)

        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
        sigma1_sq = e11 - mu1_sq
        sigma2_sq = e22 - mu2_sq
        sigma12 = e12 - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if mask is not None:
            mask = mask.to(ssim_map.dtype)
            if mask.shape[1] != ssim_map.shape[1]:
                mask = mask.expand(-1, ssim_map.shape[1], -1, -1, -1)
            return (ssim_map * mask).sum() / mask.sum().clamp(min=1.0)
        if size_average:
            return ssim_map.mean()
        return ssim_map.flatten(1).mean(1)
