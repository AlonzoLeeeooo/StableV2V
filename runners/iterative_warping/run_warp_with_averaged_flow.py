import argparse
import cv2
import torch
import os
import numpy as np
from typing import List
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from runners.iterative_warping.warp_utils import optical_flow_warping
from runners.iterative_warping.run_torch_average_flow_warping import visualize_flow

def get_average_flow_func(optical_flow, object_mask, editing_mask, channels=2, height=None, width=None):
    object_masked_flow = object_mask * optical_flow
    total_flow = object_masked_flow.sum(dim=(2, 3))
    num_pixels = object_mask.sum()
    average_flow = total_flow / num_pixels
    
    optical_flow = torch.where(editing_mask == 1, average_flow.view(1, channels, 1, 1).expand(-1, -1, height, width), optical_flow)

    return optical_flow


def images_to_gif_bytes(images: List, duration: int = 1000) -> bytes:
    with BytesIO() as output_buffer:
        images[0].save(output_buffer,
                       format='GIF',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration,
                       loop=0)
        gif_bytes = output_buffer.getvalue()
    return gif_bytes

def save_as_gif(images: List, file_path: str, duration: int = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_gif_bytes(images, duration))


def iterative_warp_with_averaged_flow(args):
    args.editing_mask = os.path.join(args.outdir, 'iterative_warping', 'editing_masks')
    args.object_mask = os.path.join(args.outdir, 'iterative_warping', 'object_masks')
    args.optical_flow = os.path.join(args.outdir, 'iterative_warping', 'optical_flows')
    
    assert os.path.exists(args.editing_mask), "Editing mask does not exist."
    assert os.path.exists(args.object_mask), "Object mask does not exist."

    # Load initial mask, i.e., editing mask
    init_mask = cv2.imread(os.path.join(args.editing_mask, f"{args.prompt.lower().replace(' ', '_')}.png"), cv2.IMREAD_GRAYSCALE)

    # Load optical flows
    optical_flow_paths = sorted(os.listdir(args.optical_flow))
    optical_flows = []
    for optical_flow_path in optical_flow_paths[:args.n_sample_frames]:
        optical_flow = np.load(os.path.join(args.optical_flow, optical_flow_path))
        optical_flow = torch.from_numpy(optical_flow)
        optical_flows.append(optical_flow)

    _, C, H, W = optical_flows[0].shape

    # Resize initial mask
    init_mask = cv2.resize(init_mask, (W, H))
    init_mask = init_mask.astype(np.float32) / 255.0
    init_mask = torch.from_numpy(init_mask).unsqueeze(0).unsqueeze(0)
    init_mask[init_mask > 0.5] = 1
    init_mask[init_mask <= 0.5] = 0

    # Create dilated mask
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)  # You can adjust the kernel size as needed
    dilated_mask = cv2.dilate(init_mask.squeeze().cpu().numpy(), kernel, iterations=args.dilation_iteration)
    
    # Erode the dilated mask
    erode_kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)  # Adjust kernel size as needed
    dilated_mask = cv2.erode(dilated_mask, erode_kernel, iterations=6)

    dilated_mask = torch.from_numpy(dilated_mask).unsqueeze(0).unsqueeze(0)
    dilated_mask[dilated_mask > 0.5] = 1
    dilated_mask[dilated_mask <= 0.5] = 0


    # Load object masks
    object_mask_paths = sorted(os.listdir(args.object_mask))
    object_masks = []
    for object_mask_path in object_mask_paths[:args.n_sample_frames]:
        object_mask = cv2.imread(os.path.join(args.object_mask, object_mask_path), cv2.IMREAD_GRAYSCALE)
        object_mask = cv2.resize(object_mask, (W, H))
        object_mask = torch.from_numpy(object_mask.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        object_mask[object_mask > 0.5] = 1
        object_mask[object_mask <= 0.5] = 0
        object_masks.append(object_mask)

    editing_masks = [dilated_mask]
    current_masks = [init_mask]

    first_dilated_editing_mask = (dilated_mask.squeeze().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, 'iterative_warping', 'warped_editing_masks', f'00000.png'), first_dilated_editing_mask)
    first_editing_mask = (init_mask.squeeze().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, 'iterative_warping', 'warped_masks', f'00000.png'), first_editing_mask)


    os.makedirs(os.path.join(args.outdir, 'iterative_warping', 'warped_editing_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'iterative_warping', 'warped_masks'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'iterative_warping', 'averaged_flows'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'iterative_warping', 'gif_outputs'), exist_ok=True)

    for i in tqdm(range(len(optical_flows))[:args.n_sample_frames]):
        # 1. Generate new optical flow
        averaged_flow = get_average_flow_func(optical_flows[i], object_masks[i], editing_masks[-1], channels=C, height=H, width=W)
        visualize_flow(averaged_flow.squeeze(0), os.path.join(args.outdir, 'iterative_warping', 'averaged_flows'), i)

        # 2. Warp the current frame and mask
        current_mask_to_warp = current_masks[-1]
        if len(current_mask_to_warp.shape) == 3:
            current_mask_to_warp = current_mask_to_warp.unsqueeze(0)

        current_editing_mask = optical_flow_warping(editing_masks[-1], averaged_flow)[0]
        current_editing_mask = current_editing_mask * editing_masks[-1]
        if len(current_editing_mask.shape) == 3:
            current_editing_mask = current_editing_mask.unsqueeze(0)
        current_editing_mask[current_editing_mask > 0.5] = 1
        current_editing_mask[current_editing_mask <= 0.5] = 0
        editing_masks.append(current_editing_mask)
        
        current_mask = optical_flow_warping(current_masks[-1], averaged_flow)[0]
        current_mask[current_mask > 0.5] = 1
        current_mask[current_mask <= 0.5] = 0
        current_mask = current_mask * current_masks[-1]
        if len(current_mask.shape) == 3:
            current_mask = current_mask.unsqueeze(0)
        current_masks.append(current_mask)

        # Convert back to numpy arrays
        current_mask = (current_mask.squeeze().numpy() * 255).astype(np.uint8)
        current_editing_mask = (current_editing_mask.squeeze().numpy() * 255).astype(np.uint8)
      
        # Save warped mask
        cv2.imwrite(os.path.join(args.outdir, 'iterative_warping', 'warped_editing_masks', f'{(i+1):05d}.png'), current_editing_mask)
        cv2.imwrite(os.path.join(args.outdir, 'iterative_warping', 'warped_masks', f'{(i+1):05d}.png'), current_mask)

    # Save results as GIF
    pil_editing_masks = [Image.fromarray((mask.squeeze().cpu().numpy() * 255).astype(np.uint8)) for mask in editing_masks]
    pil_warped_masks = [Image.fromarray((mask.squeeze().cpu().numpy() * 255).astype(np.uint8)) for mask in current_masks]
    save_as_gif(pil_editing_masks, os.path.join(args.outdir, 'iterative_warping', 'gif_outputs', 'editing_masks.gif'), duration=1000 // 16)
    save_as_gif(pil_warped_masks, os.path.join(args.outdir, 'iterative_warping', 'gif_outputs', 'warped_masks.gif'), duration=1000 // 16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--object-mask', type=str, required=True, help='Complete object masks of the source video frames')
    parser.add_argument('--optical-flow', type=str, required=True, help='Complete optical flows of the source video frames')
    parser.add_argument('--editing-mask', type=str, required=True, help='Editing mask of the first video frame')
    parser.add_argument('--outdir', type=str, default='warped-outputs')
    args = parser.parse_args()

    iterative_warp_with_averaged_flow(args)
