import os
import cv2
import numpy as np
import torch
import PIL
from PIL import Image
from models.depth_completion_net.rfc_net import RecurrentDepthCompleteNet
from utils.utils import binarize_tensor, save_as_gif

device = 'cuda' if torch.cuda.is_available() else "cpu"

def flow_completion_net_inference_runner():
    pass


def depth_completion_net_inference_runner(args):
    # 0. Define inputs
    input_depth_maps_dir = os.path.join(args.outdir, 'iterative_warping', 'averaged_depths')
    object_masks_dir = os.path.join(args.outdir, 'iterative_warping', 'object_masks')
    editing_masks_dir = os.path.join(args.outdir, 'iterative_warping', 'warped_masks')
    input_shape_path = os.path.join(args.outdir, 'iterative_warping', 'editing_masks', args.prompt.lower().replace(' ', '_') + '.png')

    # 1. Define the models
    completion_net = RecurrentDepthCompleteNet(in_channels=5, out_channels=3)
    completion_net.load_state_dict(torch.load(args.completion_net_checkpoint_path, map_location='cpu'))
    completion_net = completion_net.to(device)
    completion_net.eval()
    
    # 2. Load input data
    # 2.1 Input depth maps
    depth_path_list = sorted(os.listdir(input_depth_maps_dir))[:args.n_sample_frames]
    input_depth_maps = []
    for depth_path in depth_path_list:
        depth_map = cv2.imread(os.path.join(input_depth_maps_dir, depth_path))
        depth_map = cv2.resize(depth_map, (args.width, args.height))
        depth_map = torch.from_numpy(depth_map.astype(np.float32) / 255.).permute(2, 0, 1)
        input_depth_maps.append(depth_map)
    depth_maps_tensor = torch.stack(input_depth_maps, dim=0)
    depth_maps_tensor = depth_maps_tensor.unsqueeze(0)
    
    
    # 2.2 Input shape guidance
    shape = cv2.imread(input_shape_path)
    shape = cv2.resize(shape, (args.width, args.height))
    shape_tensor = torch.from_numpy(shape.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)
    shape_tensor = torch.sum(shape_tensor / 3, dim=1, keepdim=True)
    shape_tensor = binarize_tensor(shape_tensor)
    shape_tensor = torch.stack([shape_tensor] * len(input_depth_maps), dim=1)
    
    # 2.3 Input editing masks
    mask_list = sorted(os.listdir(object_masks_dir))[:args.n_sample_frames]
    input_masks = []
    for mask_path in mask_list:
        mask = cv2.imread(os.path.join(object_masks_dir, mask_path))
        mask = cv2.resize(mask, (args.width, args.height))

        # Dilate the mask, make sure to cover the undesired regions
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        mask = torch.from_numpy(mask.astype(np.float32) / 255.).permute(2, 0, 1)
        mask = torch.sum(mask / 3, dim=0, keepdim=True)
        input_masks.append(mask)
    mask_tensor = torch.stack(input_masks, dim=0)
    mask_tensor = binarize_tensor(mask_tensor)
    mask_tensor = mask_tensor.unsqueeze(0)
    
    # 2.4 Input object masks
    warped_mask_list = sorted(os.listdir(editing_masks_dir))[:args.n_sample_frames]
    input_warped_masks = []
    for warped_mask_path in warped_mask_list:
        warped_mask = cv2.imread(os.path.join(editing_masks_dir, warped_mask_path))
        warped_mask = cv2.resize(warped_mask, (args.width, args.height))
        warped_mask = torch.from_numpy(warped_mask.astype(np.float32) / 255.).permute(2, 0, 1)
        warped_mask = torch.sum(warped_mask / 3, dim=0, keepdim=True)
        input_warped_masks.append(warped_mask)
    warped_mask_tensor = torch.stack(input_warped_masks, dim=0)
    warped_mask_tensor = binarize_tensor(warped_mask_tensor)
    warped_mask_tensor = warped_mask_tensor.unsqueeze(0)
    
    mask_tensor = (1 - warped_mask_tensor) * mask_tensor

    # Save mask_tensor
    output_mask_path = os.path.join(args.outdir, 'iterative_warping', 'completion_net_mask_region')
    os.makedirs(output_mask_path, exist_ok=True)
    for i in range(mask_tensor.size(1)):
        mask = (mask_tensor[:, i, :, :, :].squeeze(0).permute(1, 2, 0) * 255.).cpu().numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(output_mask_path, f"{i:05d}.png"), np.clip(mask, a_min=0, a_max=255))
    
    # 3. Forward
    with torch.no_grad():
        pred_depth_maps, _ = completion_net(depth_maps_tensor.to(device), mask_tensor.to(device), shape_tensor.to(device))
    pred_depth_maps = pred_depth_maps.to(device)
    pred_depth_maps = (1 - mask_tensor) * depth_maps_tensor + mask_tensor * pred_depth_maps
    
    # 4. Visualization
    # 4.1 Depth maps
    vis_depth_maps = []
    for i in range(pred_depth_maps.size(1)):
       vis_depth_map = (pred_depth_maps[:, i, :, :, :].squeeze(0).permute(1, 2, 0) * 255.).cpu().numpy().astype(np.uint8)
       vis_depth_maps.append(vis_depth_map)
    
       
    # 4.3 Save everything
    output_depth_path = os.path.join(args.outdir, 'iterative_warping', 'final_depth_maps')
    os.makedirs(output_depth_path, exist_ok=True)
    for index in range(len(vis_depth_maps)):
        cv2.imwrite(os.path.join(output_depth_path, f"{index:05d}.png"), np.clip(vis_depth_maps[index], a_min=0, a_max=255))
    pil_vis_depth_maps = []
    for vis_depth_map in vis_depth_maps:
        pil_vis_depth_maps.append(Image.fromarray(vis_depth_map))
    save_as_gif(pil_vis_depth_maps, os.path.join(args.outdir, 'iterative_warping', 'gif_outputs', 'final_depth_maps.gif'), duration=125)
    