import argparse
import cv2
import torch
import os
import numpy as np
from tqdm import tqdm

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(img.astype(np.float32) / 255.).unsqueeze(0).unsqueeze(0)

def get_averaged_depths_main_func(args):
    depth_dir = os.path.join(args.outdir, 'iterative_warping', 'depth_maps')
    output_dir = os.path.join(args.outdir, 'iterative_warping', 'averaged_depths')
    object_mask_dir = os.path.join(args.outdir, 'iterative_warping', 'object_masks')
    editing_mask_dir = os.path.join(args.outdir, 'iterative_warping', 'warped_masks')
    os.makedirs(output_dir, exist_ok=True)

    # Get sorted lists of all files in each input directory
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.png') or f.endswith('.jpg')])[:args.n_sample_frames]
    object_mask_files = sorted([f for f in os.listdir(object_mask_dir) if f.endswith('.png') or f.endswith('.jpg')])[:args.n_sample_frames]
    editing_mask_files = sorted([f for f in os.listdir(editing_mask_dir) if f.endswith('.png') or f.endswith('.jpg')])[:args.n_sample_frames]

    for i, (depth_file, object_mask_file, editing_mask_file) in enumerate(tqdm(zip(depth_files, object_mask_files, editing_mask_files), total=len(depth_files))):
        # 1. Load depth map
        depth_map = load_image(os.path.join(depth_dir, depth_file))
        _, _, H, W = depth_map.shape
        
        # 2. Load object mask
        object_mask = load_image(os.path.join(object_mask_dir, object_mask_file))
        object_mask = cv2.resize(object_mask.squeeze().numpy(), (W, H))
        object_mask = torch.from_numpy(object_mask).unsqueeze(0).unsqueeze(0)
        object_mask[object_mask > 0.5] = 1
        object_mask[object_mask <= 0.5] = 0

        # 3. Load editing mask
        editing_mask = load_image(os.path.join(editing_mask_dir, editing_mask_file))
        editing_mask = cv2.resize(editing_mask.squeeze().numpy(), (W, H))
        editing_mask = torch.from_numpy(editing_mask).unsqueeze(0).unsqueeze(0)
        editing_mask[editing_mask > 0.5] = 1
        editing_mask[editing_mask <= 0.5] = 0

        # 4. Compute average depth within the object mask
        object_masked_depth = object_mask * depth_map
        total_depth = object_masked_depth.sum()
        num_pixels = object_mask.sum()
        average_depth = total_depth / num_pixels
        
        # 5. Apply average depth to depths within the editing mask
        averaged_depth_map = torch.where(editing_mask == 1, average_depth, depth_map)

        # 6. Save the result
        output_depth = (averaged_depth_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
        output_filename = f'{i:05d}.png'
        cv2.imwrite(os.path.join(output_dir, output_filename), output_depth)
