# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import os
import cv2
import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)



def visualize_flow(flow, outdir, index, save_image_flow_concat=False):
    flow = flow.permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flow = flow_to_image(flow)
    
    os.makedirs(os.path.join(outdir, 'optical-flow-frames'), exist_ok=True)
    cv2.imwrite(os.path.join(outdir, 'optical-flow-frames', f'{index:05d}.png'), flow[:, :, [2,1,0]])
    

def resize_flow(flow, newh, neww):
    oldh, oldw = flow.shape[0:2]
    flow = cv2.resize(flow, (neww, newh), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= neww / oldw
    flow[:, :, 1] *= newh / oldh
    return flow


def dequantize(arr, min_val, max_val, levels, dtype=np.float64):
    """Dequantize an array.

    Args:
        arr (ndarray): Input array.
        min_val (scalar): Minimum value to be clipped.
        max_val (scalar): Maximum value to be clipped.
        levels (int): Quantization levels.
        dtype (np.type): The type of the dequantized array.

    Returns:
        tuple: Dequantized array.
    """
    if not (isinstance(levels, int) and levels > 1):
        raise ValueError(f'levels must be a positive integer, but got {levels}')
    if min_val >= max_val:
        raise ValueError(f'min_val ({min_val}) must be smaller than max_val ({max_val})')

    dequantized_arr = (arr + 0.5).astype(dtype) * (max_val - min_val) / levels + min_val

    return dequantized_arr


def dequantize_flow(dx, dy, max_val=0.02, denorm=True):
    """Recover from quantized flow.

    Args:
        dx (ndarray): Quantized dx.
        dy (ndarray): Quantized dy.
        max_val (float): Maximum value used when quantizing.
        denorm (bool): Whether to multiply flow values with width/height.

    Returns:
        ndarray: Dequantized flow.
    """
    assert dx.shape == dy.shape
    assert dx.ndim == 2 or (dx.ndim == 3 and dx.shape[-1] == 1)

    dx, dy = [dequantize(d, -max_val, max_val, 255) for d in [dx, dy]]

    if denorm:
        dx *= dx.shape[1]
        dy *= dx.shape[0]
    flow = np.dstack((dx, dy))
    return flow


def flowread(flow_path, quantize=False, concat_axis=0, *args, **kwargs):
    """Read an optical flow map.

    Args:
        flow_path (ndarray or str): Flow path.
        quantize (bool): whether to read quantized pair, if set to True,
            remaining args will be passed to :func:`dequantize_flow`.
        concat_axis (int): The axis that dx and dy are concatenated,
            can be either 0 or 1. Ignored if quantize is False.

    Returns:
        ndarray: Optical flow represented as a (h, w, 2) numpy array
    """
    if quantize:
        assert concat_axis in [0, 1]
        cat_flow = cv2.imread(flow_path, cv2.IMREAD_UNCHANGED)
        if cat_flow.ndim != 2:
            raise IOError(f'{flow_path} is not a valid quantized flow file, its dimension is {cat_flow.ndim}.')
        assert cat_flow.shape[concat_axis] % 2 == 0
        dx, dy = np.split(cat_flow, 2, axis=concat_axis)
        flow = dequantize_flow(dx, dy, *args, **kwargs)
    else:
        with open(flow_path, 'rb') as f:
            try:
                header = f.read(4).decode('utf-8')
            except Exception:
                raise IOError(f'Invalid flow file: {flow_path}')
            else:
                if header != 'PIEH':
                    raise IOError(f'Invalid flow file: {flow_path}, header does not contain PIEH')

            w = np.fromfile(f, np.int32, 1).squeeze()
            h = np.fromfile(f, np.int32, 1).squeeze()
            # flow = np.fromfile(f, np.float32, w * h * 2).reshape((h, w, 2))
            flow = np.fromfile(f, np.float16, w * h * 2).reshape((h, w, 2))

    return flow.astype(np.float32)


# Flow visualization code adapted from https://github.com/tomrunia/OpticalFlow_Visualization
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732

@torch.no_grad()
def flow_to_image(flow: torch.Tensor) -> torch.Tensor:

    """
    Converts a flow to an RGB image.

    Args:
        flow (Tensor): Flow of shape (N, 2, H, W) or (2, H, W) and dtype torch.float.

    Returns:
        img (Tensor): Image Tensor of dtype uint8 where each color corresponds
            to a given flow direction. Shape is (N, 3, H, W) or (3, H, W) depending on the input.
    """

    if flow.dtype != torch.float:
        raise ValueError(f"Flow should be of dtype torch.float, got {flow.dtype}.")

    orig_shape = flow.shape
    if flow.ndim == 3:
        flow = flow[None]  # Add batch dim

    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError(f"Input flow should have shape (2, H, W) or (N, 2, H, W), got {orig_shape}.")

    max_norm = torch.sum(flow**2, dim=1).sqrt().max()
    epsilon = torch.finfo((flow).dtype).eps
    normalized_flow = flow / (max_norm + epsilon)
    img = _normalized_flow_to_image(normalized_flow)

    if len(orig_shape) == 3:
        img = img[0]  # Remove batch dim
    return img

@torch.no_grad()
def _normalized_flow_to_image(normalized_flow: torch.Tensor) -> torch.Tensor:

    """
    Converts a batch of normalized flow to an RGB image.

    Args:
        normalized_flow (torch.Tensor): Normalized flow tensor of shape (N, 2, H, W)
    Returns:
       img (Tensor(N, 3, H, W)): Flow visualization image of dtype uint8.
    """

    N, _, H, W = normalized_flow.shape
    device = normalized_flow.device
    flow_image = torch.zeros((N, 3, H, W), dtype=torch.uint8, device=device)
    colorwheel = _make_colorwheel().to(device)  # shape [55x3]
    num_cols = colorwheel.shape[0]
    norm = torch.sum(normalized_flow**2, dim=1).sqrt()
    a = torch.atan2(-normalized_flow[:, 1, :, :], -normalized_flow[:, 0, :, :]) / torch.pi
    fk = (a + 1) / 2 * (num_cols - 1)
    k0 = torch.floor(fk).to(torch.long)
    k1 = k0 + 1
    k1[k1 == num_cols] = 0
    f = fk - k0

    for c in range(colorwheel.shape[1]):
        tmp = colorwheel[:, c]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        col = 1 - norm * (1 - col)
        flow_image[:, c, :, :] = torch.floor(255. * col)
    return flow_image


@torch.no_grad()
def _make_colorwheel() -> torch.Tensor:
    """
    Generates a color wheel for optical flow visualization as presented in:
    Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
    URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf.

    Returns:
        colorwheel (Tensor[55, 3]): Colorwheel Tensor.
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = torch.floor(255. * torch.arange(0., RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - torch.floor(255. * torch.arange(0., YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = torch.floor(255. * torch.arange(0., GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - torch.floor(255. * torch.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = torch.floor(255. * torch.arange(0., BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - torch.floor(255. * torch.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel
