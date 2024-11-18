import os
import torch
from PIL import Image
from tqdm import tqdm
from models.midas.midas import DepthMidas

device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'

def midas_depth_estimation_runner(args):
    depth_dir = os.path.join(args.outdir, 'iterative_warping', 'depth_maps')
    os.makedirs(depth_dir, exist_ok=True)

    depth_estimator = DepthMidas(model_path=args.midas_checkpoint_path, device=device)

    # Get all image files
    image_files = [f for f in sorted(os.listdir(args.source_video_frames))[:args.n_sample_frames] if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    progress_bar = tqdm(total=len(image_files))
    for image_file in image_files:
        progress_bar.update(1)
        
        # Load image
        image_path = os.path.join(args.source_video_frames, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # Estimate depth
        depth = depth_estimator.estimate([image])[0]
        
        # Save depth map
        output_path = os.path.join(depth_dir, f"{image_file}")
        depth.save(output_path)

    progress_bar.close()

