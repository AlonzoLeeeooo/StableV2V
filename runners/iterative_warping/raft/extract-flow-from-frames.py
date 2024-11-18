import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, outdir, index):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imwrite(os.path.join(outdir, 'visualization', f'{index:05d}.png'), img_flo[:, :, [2,1,0]]) # /255.0
  

def demo(args):
    # 0. Define RAFT model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # 1. Load in frames path
    frames_path = []
    for path in sorted(os.listdir(args.path)):
        frames_path.append(os.path.join(args.path, path))
    

    # 2. Start extracting optical flows
    with torch.no_grad():
        for index in range(len(frames_path)):
            if index + 1 < len(frames_path):
                image1 = load_image(frames_path[index + 1])
                image2 = load_image(frames_path[index])

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                np.save(os.path.join(args.outdir, 'flow-up', f'{index:05d}'), flow_up.cpu())
                np.save(os.path.join(args.outdir, 'flow-low', f'{index:05d}'), flow_low.cpu())
                viz(image1, flow_up, args.outdir, index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/Users/liuchang/Desktop/Workspaces/checkpoints/raft/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--path', type=str, help='path of video frames')
    parser.add_argument('--outdir', type=str, default='outputs', help='output directory')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'visualization'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'flow-up'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'flow-low'), exist_ok=True)
    demo(args)
