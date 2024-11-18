import os
import torch
import sys

import numpy as np
from PIL import Image

from models.u2net.u2net import U2NET # full size version 173.6 MB
from models.u2net.u2net import U2NETP # small version u2net 4.7 MB

import decord
from einops import rearrange
from tqdm import tqdm

# funciton to return a path list from a directory
def get_files(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for filepath in files:
            file_list.append(os.path.join(root, filepath))

    return file_list

# function to return a path list from a txt file
def get_files_from_txt(path):
    file_list = []
    f = open(path)
    for line in f.readlines():
        line = line.strip("\n")
        file_list.append(line)
        sys.stdout.flush()
    f.close()

    return file_list

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(filename, pred, outdir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')

    im.save(os.path.join(outdir, filename + '.png'))

def main():

    # 1. Define basic configurations
    model_name = 'u2net'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-root', type=str, default='')
    parser.add_argument('--model-dir', default='', type=str, help='Pre-trained model weights of U2-Net')
    parser.add_argument('--outdir', default='', type=str, help='Output path of extracted mask of first frames')
    parser.add_argument('--size', default=256, type=int)

    args = parser.parse_args()

    prediction_dir = args.outdir

    if not prediction_dir.endswith('/'):
        prediction_dir = prediction_dir + '/'
    model_dir = args.model_dir


    # 2. Define model
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    
    
    video_paths = os.listdir(args.video_root)
    progress_bar = tqdm(total=len(video_paths))
    for video_path in video_paths:
        progress_bar.update(1)
        os.makedirs(os.path.join(args.outdir, video_path), exist_ok=True)
        frame_paths = os.listdir(os.path.join(args.video_root, video_path))
        for frame_path in frame_paths:
            video_frame = Image.open(os.path.join(args.video_root, video_path, frame_path)).resize((args.size, args.size))
            filename = frame_path.split('.')[0]
            video_frame = torch.from_numpy(np.array(video_frame).astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
            
            d1,d2,d3,d4,d5,d6,d7= net(video_frame)
            pred = d1[:,0,:,:]
            pred = normPRED(pred)

            prediction_dir = os.path.join(args.outdir, video_path)
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(filename, pred, prediction_dir)

            del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()