from tinygrad import Tensor, dtypes
from tinygrad.nn import BatchNorm
import numpy as np
import math
from typing import Callable, List

def flatten_forward_unflatten(fn_list: List[Callable[[Tensor], Tensor]], image_tensor: Tensor) -> Tensor:
    """Helper to temporarily flatten extra dims at the start of the image tensor.

    Args:
        fn: Callable that the image tensor will be passed to. It should accept (B, C, H, W) and return
            (B, *), where * is any number of dimensions.
        image_tensor: An image tensor of shape (**, C, H, W), where ** is any number of dimensions, generally
            different from *.
    Returns:
        A return value from the callable reshaped to (**, *).
    """
    if image_tensor.ndim == 4:
        print(f'image_tensor.ndim = 4. shape: {image_tensor.shape}')
        print(f'fn_list: {fn_list}')
        return image_tensor.sequential(fn_list)
    start_dims = image_tensor.shape[:-3]
    inp = image_tensor.flatten(end_dim=-4)
    if len(fn_list) == 1:
        flat_out = fn_list[0](inp)
        return flat_out.reshape((*start_dims, *flat_out.shape[1:]))
    else:
        flat_out = inp.sequential(fn_list)
        return flat_out.reshape((*start_dims, *flat_out.shape[1:]))

def orthogonal_(tensor: Tensor, gain=1.0):
    if tensor.ndim < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    if tensor.numel() == 0:
        # no-op
        return tensor

    rows = tensor.shape[0]
    cols = tensor.numel() // rows
    flattened = np.random.normal(0, 1, (rows, cols))

    if rows < cols:
        flattened = flattened.T

    # Compute the qr factorization
    q, r = np.linalg.qr(flattened)

    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = np.diag(r)
    ph = np.sign(d)
    q *= ph

    if rows < cols:
        q = q.T

    final_q = q.reshape(tensor.shape)
    final_q *= gain

    return Tensor(final_q, dtype=tensor.dtype, requires_grad=False)

def calculate_gain(nonlinearity: str, param: float = None) -> float:
    """
    Return the recommended gain value for the given nonlinearity function.
    """
    linear_gains = {'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'}
    
    if nonlinearity in linear_gains:
        return 1.0
    elif nonlinearity == 'sigmoid':
        return 1.0
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            param = 0.01  # Default negative_slope
        return math.sqrt(2.0 / (1 + param ** 2))
    elif nonlinearity == 'selu':
        return 3.0 / 4  # Note: Value used in the paper, but not the same as PyTorch
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")

def linspace(start, end, steps, dtype=None):
    if dtype is None:
        dtype = dtypes.float32
    
    # Create a tensor with values from 0 to steps-1
    t = Tensor.arange(steps, dtype=dtype)
    
    # Scale and shift the values
    step_size = (end - start) / (steps - 1)
    return t * step_size + start

def grid_sample_4d(input: Tensor, grid: Tensor, mode='bilinear', padding_mode='zeros', align_corners=False) -> Tensor:
    B, C, H, W = input.shape
    _, H_out, W_out, _ = grid.shape
    
    # Rescale grid coordinates to match input dimensions
    if align_corners:
        x = (grid[:, :, :, 0] + 1) * (W - 1) / 2
        y = (grid[:, :, :, 1] + 1) * (H - 1) / 2
    else:
        x = ((grid[:, :, :, 0] + 1) * W - 1) / 2
        y = ((grid[:, :, :, 1] + 1) * H - 1) / 2
    
    # Get integer parts of coordinates
    x0 = x.floor()
    x1 = x0 + 1
    y0 = y.floor()
    y1 = y0 + 1
    
    # Clip coordinates to valid range
    x0 = x0.clip(0, W - 1)
    x1 = x1.clip(0, W - 1)
    y0 = y0.clip(0, H - 1)
    y1 = y1.clip(0, H - 1)
    
    # Compute weights
    wa = (1 - (x - x0).abs()) * (1 - (y - y0).abs())
    wb = (1 - (x - x0).abs()) * (1 - (y - y1).abs())
    wc = (1 - (x - x1).abs()) * (1 - (y - y0).abs())
    wd = (1 - (x - x1).abs()) * (1 - (y - y1).abs())
    
    # Convert to integer indices
    x0 = x0.int()
    x1 = x1.int()
    y0 = y0.int()
    y1 = y1.int()
    # Reshape input and indices for matmul
    input_flat = input.reshape(B, C, H * W)
    x0_flat = x0.reshape(B, 1, -1)
    x1_flat = x1.reshape(B, 1, -1)
    y0_flat = y0.reshape(B, 1, -1)
    y1_flat = y1.reshape(B, 1, -1)
    
    # Create index tensors
    idx_a = y0_flat * W + x0_flat
    idx_b = y1_flat * W + x0_flat
    idx_c = y0_flat * W + x1_flat
    idx_d = y1_flat * W + x1_flat
    
    # Create selection matrices
    sel_a = (Tensor.arange(H * W).reshape(1, 1, -1) == idx_a).float()
    sel_b = (Tensor.arange(H * W).reshape(1, 1, -1) == idx_b).float()
    sel_c = (Tensor.arange(H * W).reshape(1, 1, -1) == idx_c).float()
    sel_d = (Tensor.arange(H * W).reshape(1, 1, -1) == idx_d).float()
    
    # Perform interpolation using matmul
    Ia = input_flat @ (wa.reshape(B, 1, -1) * sel_a)
    Ib = input_flat @ (wb.reshape(B, 1, -1) * sel_b)
    Ic = input_flat @ (wc.reshape(B, 1, -1) * sel_c)
    Id = input_flat @ (wd.reshape(B, 1, -1) * sel_d)
    
    out = (Ia + Ib + Ic + Id).reshape(B, C, H_out, W_out)
    
    # Apply zero padding if required
    if padding_mode == 'zeros':
        mask = ((x >= 0).int() & (x <= W - 1).int() & (y >= 0).int() & (y <= H - 1).int()).float()
        out = out * mask.reshape(B, 1, H_out, W_out)
    
    return out

def random_shifts_aug(x: Tensor, max_random_shift_ratio: float) -> Tensor:
    """Randomly shifts images horizontally and vertically."""
    b, c, h, w = x.shape
    assert h == w, "non-square images not handled yet"
    pad = int(round(max_random_shift_ratio * h))
    
    # Pad the input tensor
    x_padded = x.pad(((0, 0), (0, 0), (pad, pad), (pad, pad)))
    
    # Generate random shifts for each image in the batch
    shift_x = (Tensor.rand(b) * (2 * pad)).cast(dtype=dtypes.int)
    shift_y = (Tensor.rand(b) * (2 * pad)).cast(dtype=dtypes.int)

    # Initialize an empty tensor to store shifted results
    shifted = Tensor.zeros((b, c, h, w)).contiguous()

    # Apply the shifts by slicing the padded tensor
    print(f'applying shifts through {b} many slices')
    for i in range(b):
        print(f'shift_x[i]: {shift_x[i]}, shift_y[i]: {shift_y[i]}')
        slice_x = slice(shift_x[i].item(), shift_x[i].item() + h)
        slice_y = slice(shift_y[i].item(), shift_y[i].item() + w)
        shifted[i] = x_padded[i, :, slice_x, slice_y].contiguous()

    return shifted

def random_shifts_aug_2(x: Tensor, max_random_shift_ratio: float) -> Tensor:
    """Randomly shifts images horizontally and vertically.

    Adapted from https://github.com/facebookresearch/drqv2
    """
    b, _, h, w = x.size()
    assert h == w, "non-square images not handled yet"
    pad = int(round(max_random_shift_ratio * h))
    padding = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x = x.pad(padding)
    eps = 1.0 / (h + 2 * pad)
    arange =  Tensor.arange(-1.0 + eps, 1.0 - eps, eps)[:h]
    arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    base_grid = arange.cat(arange.transpose(1, 0), dim=2)
    base_grid = base_grid.unsqueeze(0).repeat(b, 1, 1, 1)
    # A random shift in units of pixels and within the boundaries of the padding.
    shift = Tensor.randint(
        *(b, 1, 1, 2), # shape
        low=0,
        high=2 * pad + 1,
    ).cast(dtype=dtypes.float32)
    shift *= 2.0 / (h + 2 * pad)
    grid = base_grid + shift
    print(f'random_shifts_aug x.shape: {x.shape}, grid.shape: {grid.shape}')
    return grid_sample_4d(x, grid, padding_mode="zeros", align_corners=False)
