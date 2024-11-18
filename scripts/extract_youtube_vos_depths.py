import os
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from models.midas.midas import DepthMidas

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--midas-path', type=str, default='', help='Path to MiDaS model weights')
parser.add_argument('--dataset-path', type=str, default='input_image_examples', help='Path to input image dataset')
parser.add_argument('--outdir', type=str, default='output_depth_examples', help='Output directory for depth maps')

args = parser.parse_args()

device = args.device
midas_path = args.midas_path
dataset_path = args.dataset_path
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

depth_estimator = DepthMidas(model_path=midas_path,
                             device=device)
video_paths = os.listdir(dataset_path)
progress_bar = tqdm(total=len(video_paths))
for video_path in video_paths:
    progress_bar.update(1)
    os.makedirs(os.path.join(outdir, video_path), exist_ok=True)
    frame_paths = os.listdir(os.path.join(dataset_path, video_path))
    frames_pil = []
    frame_name_list = []
    for frame_path in frame_paths:
        video_frame = Image.open(os.path.join(dataset_path, video_path, frame_path))
        frames_pil.append(video_frame)
        frame_name_list.append(frame_path.split('.')[0])
    depths_pil = depth_estimator.estimate(frames_pil)
    for depth, frame_name in zip(depths_pil, frame_name_list):
        depth.save(os.path.join(outdir, video_path, frame_name + '.png'))