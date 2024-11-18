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


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imwrite(f'outputs/visualization.png', img_flo[:, :, [2,1,0]]) # /255.0
  

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model, map_location='cpu'))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        image1 = load_image(args.image1)
        image2 = load_image(args.image2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        np.save(os.path.join('outputs', 'flow-up'), flow_up.cpu())
        np.save(os.path.join('outputs', 'flow-low'), flow_low.cpu())
        viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/Users/liuchang/Desktop/Workspaces/checkpoints/raft/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--image1', type=str, help='the first image')
    parser.add_argument('--image2', type=str, help='the second image')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)
    demo(args)
