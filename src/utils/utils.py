import math
import torch
import argparse
import yaml
import random
import numpy as np

# helpers functions

def load_config_from_yaml(config_file_path):
    with open(config_file_path, 'r') as f:
        config_data = yaml.safe_load(f)

    args = argparse.Namespace(**config_data)
    return args

#learning rate scheduler
def build_scheduler(opt, kind, total_steps, warmup_steps=0, min_lr_ratio=0.1,
                    step_size=20000, gamma=0.5, patience=20):
    """kind: none | cosine | warmup_cosine | step | plateau
    'plateau' no se llama por iteración: se llama con la métrica de validación."""
    if kind == 'none':
        return None

    if kind == 'cosine':
        base_lr = opt.param_groups[0]['lr']
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps, eta_min=base_lr * min_lr_ratio)

    if kind == 'warmup_cosine':
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            cos = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cos
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if kind == 'step':
        return torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    if kind == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=gamma, patience=patience,
            min_lr=opt.param_groups[0]['lr'] * min_lr_ratio)

    raise ValueError(f'lr_scheduler desconocido: {kind}')


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def cycle_pair(dl):
    while True:
        for mri_data, pet_data in dl:
            yield mri_data, pet_data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor): #(4,8)
    groups = num // divisor # 0
    remainder = num % divisor # 4
    arr = [divisor] * groups # [0]
    if remainder > 0:
        arr.append(remainder) # [0,4]
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def print_model_size(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def set_seed_everywhere(seed):
    """Set seed for reproducibility."""
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def positional_encoding_2d(x, h, w):
    B = x.shape[0]
    
    # Generate row and column indices
    row_indices = torch.linspace(0, 1, h).view(1, -1, 1).expand(B, -1, -1).to(x.device)
    col_indices = torch.linspace(0, 1, w).view(1, 1, -1).expand(B, -1, -1).to(x.device)
    
    # Reshape x for broadcasting
    x = x.view(B, 1, 1)
    
    # Create sinusoidal patterns modulated by the input value batch
    row_pattern = torch.sin(2 * 3.14159 * x * 10 * row_indices)
    col_pattern = torch.sin(2 * 3.14159 * x * 10 * col_indices)

    # Combine row and column patterns
    encoded = row_pattern + col_pattern

    return encoded
