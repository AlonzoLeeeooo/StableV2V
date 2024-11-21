import sys
sys.path.append('core')

import os
import cv2
import numpy as np
import torch
from PIL import Image

from models.raft.raft import RAFT
from runners.iterative_warping.raft.core.utils.flow_viz import flow_to_image
from runners.iterative_warping.raft.core.utils.utils import InputPadder



device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.resize(img, (512, 512))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)


def viz(img, flo, outdir, index):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imwrite(os.path.join(outdir, 'visualization', f'{index:05d}.png'), img_flo[:, :, [2,1,0]]) # /255.0
  

def raft_flow_extraction_runner(args):
    # 0. Define RAFT model
    model = torch.nn.DataParallel(RAFT())
    model.load_state_dict(torch.load(args.raft_checkpoint_path, map_location='cpu'))

    model = model.module
    model.to(device)
    model.eval()
    os.makedirs(os.path.join(args.outdir, 'iterative_warping', 'optical_flows'), exist_ok=True)

    # 1. Load in frames path
    frames_path = []
    for path in sorted(os.listdir(args.source_video_frames)):
        frames_path.append(os.path.join(args.source_video_frames, path))
    

    # 2. Start extracting optical flows
    with torch.no_grad():
        for index in range(min(args.n_sample_frames, len(frames_path))):
            if index + 1 < len(frames_path):
                image1 = load_image(frames_path[index + 1])
                image2 = load_image(frames_path[index])

                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                np.save(os.path.join(args.outdir, 'iterative_warping', 'optical_flows', f'{index:05d}'), flow_up.cpu())
                