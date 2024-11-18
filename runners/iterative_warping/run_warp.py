import argparse
import cv2
import torch
import os
import numpy as np
from typing import List
from io import BytesIO
from runners.iterative_warping.warp_utils import optical_flow_warping


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

def warp(init_frame_path, flows):
    
    # 2. Load in initial frame
    init_frame = cv2.imread(init_frame_path)
    init_frame = cv2.resize(init_frame, (W, H))                               # Resize to make sure that resolution is aligned
    init_frame = init_frame / 255.0
    init_frame = torch.from_numpy(init_frame).float()
    init_frame = init_frame.permute(2, 0, 1).unsqueeze(0)
    
    # 3. Warping
    warped_frames = []
    for index in range(len(optical_flows)):
        current_frame = init_frame if index == 0 else warped_frame_tensor
        if len(current_frame.shape) == 3:
            current_frame = current_frame.unsqueeze(0)
        warped_frame_tensor = optical_flow_warping(current_frame, optical_flows[index])[0]
        warped_frame = warped_frame_tensor.permute(1, 2, 0).numpy()
        warped_frames.append(warped_frame * 255)
        cv2.imwrite(os.path.join(args.outdir, f'{index:05d}.png'), warped_frame * 255)
        
    # TODO: 4. Save gif output
    # pil_warped_frames = []
    # for warped_frame in warped_frames:
    #     pil_warped_frame = Image.fromarray(warped_frame)
    #     pil_warped_frames.append(pil_warped_frame)
    # save_as_gif(pil_warped_frames, os.path.join(args.outdir, 'result.gif'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-frame', type=str, default='')
    parser.add_argument('--optical-flow', type=str, default='')
    parser.add_argument('--outdir', type=str, default='warped-outputs')
    args = parser.parse_args()

     # 0. Create directories
    os.makedirs(args.outdir, exist_ok=True)

    # 1. Load in pre-extracted optical flows
    optical_flow_paths = os.listdir(args.optical_flow)
    optical_flows = []
    for optical_flow_path in optical_flow_paths:
        optical_flow = np.load(os.path.join(args.optical_flow, optical_flow_path))
        # optical_flow = cv2.medianBlur(optical_flow, ksize=23)
        optical_flow = torch.from_numpy(optical_flow)
        optical_flows.append(optical_flow)
    _, C, H, W = optical_flows[0].shape


    warp(init_frame_path=args.init_frame,
         flows=optical_flows)