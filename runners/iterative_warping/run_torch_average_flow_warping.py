import argparse
import cv2
import torch
import os
import numpy as np
from typing import List
from io import BytesIO


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



def visualize_flow(flow, outdir, index):
    flow = flow.permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flow = flow_to_image(flow)
    
    os.makedirs(os.path.join(outdir), exist_ok=True)
    cv2.imwrite(os.path.join(outdir, f'{index:05d}.png'), flow[:, :, [2,1,0]])

def images_to_gif_bytes(images: List, duration: int = 1000) -> bytes:
    with BytesIO() as output_buffer:
        # Save the first image
        images[0].save(output_buffer,
                       format='GIF',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration,
                       loop=0)  # 0 means the GIF will loop indefinitely

        # Get the byte array from the buffer
        gif_bytes = output_buffer.getvalue()

    return gif_bytes


def save_as_gif(images: List, file_path: str, duration: int = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_gif_bytes(images, duration))
        
# This one is used to execute the script `torch_average_flow_warping.py`
def get_average_flow_main_func(args, return_flow=False):

    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load optical flow
    optical_flow = np.load(args.optical_flow)
    optical_flow = torch.from_numpy(optical_flow)
    _, C, H, W = optical_flow.shape
    
    # 2. Load object mask
    object_mask = cv2.imread(args.object_mask, cv2.IMREAD_GRAYSCALE)
    object_mask = cv2.resize(object_mask, (W, H))
    object_mask = torch.from_numpy(object_mask.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0)
    object_mask[object_mask > 0.5] = 1
    object_mask[object_mask <= 0.5] = 0

    # 3. Load editing mask
    editing_mask = cv2.imread(args.editing_mask, cv2.IMREAD_GRAYSCALE)
    editing_mask = cv2.resize(editing_mask, (W, H))
    # Dilate the editing mask using cv2
    kernel = np.ones((11, 11), np.uint8)  # You can adjust the kernel size as needed
    editing_mask = cv2.dilate(editing_mask, kernel, iterations=1)
    editing_mask = torch.from_numpy(editing_mask.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0)
    editing_mask[editing_mask > 0.5] = 1
    editing_mask[editing_mask <= 0.5] = 0
    


    # 4. Compute average optical flow within the object mask
    object_masked_flow = object_mask * optical_flow
    total_flow = object_masked_flow.sum(dim=(2, 3))
    num_pixels = object_mask.sum()
    average_flow = total_flow / num_pixels
    
    # 5. Apply average flow to optical flows within the editing mask
    optical_flow = torch.where(editing_mask == 1, average_flow.view(1, C, 1, 1).expand(-1, -1, H, W), optical_flow)

    if return_flow:
        return optical_flow

        # 6. Visualize the result
        visualize_flow(optical_flow.squeeze(0), args.outdir, index=0)


# This one is used for external execution
def get_average_flow_func(optical_flow, object_mask, editing_mask):
    # 4. Compute average optical flow within the object mask
    object_masked_flow = object_mask * optical_flow
    total_flow = object_masked_flow.sum(dim=(2, 3))
    num_pixels = object_mask.sum()
    average_flow = total_flow / num_pixels
    
    # 5. Apply average flow to optical flows within the editing mask
    optical_flow = torch.where(editing_mask == 1, average_flow.view(1, C, 1, 1).expand(-1, -1, H, W), optical_flow)

    return optical_flow

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-frame', type=str, default='')
    parser.add_argument('--optical-flow', type=str, default='')
    parser.add_argument('--object-mask', type=str, default='')
    parser.add_argument('--editing-mask', type=str, default='')
    parser.add_argument('--outdir', type=str, default='averaged-flow')
    args = parser.parse_args()

    get_average_flow_main_func(args)
   
