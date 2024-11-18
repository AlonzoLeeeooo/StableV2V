import os
import argparse
from PIL import Image
from tqdm import tqdm
from models.midas.midas import DepthMidas
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Extract depth maps from images")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to use for computation")
    parser.add_argument("--midas_path", default='/Users/liuchang/Desktop/Workspaces/checkpoints/dpt_swin2_large_384.pt',
                        help="Path to MiDaS model")
    parser.add_argument("--input_dir", default='inpainted_outputs',
                        help="Directory containing input images")
    parser.add_argument("--output_dir", default='experimental_scripts/output_depth_examples',
                        help="Directory to save output depth maps")
    return parser.parse_args()

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    depth_estimator = DepthMidas(model_path=args.midas_path, device=args.device)

    # Get all image files
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    progress_bar = tqdm(total=len(image_files))
    for image_file in image_files:
        progress_bar.update(1)
        
        # Load image
        image_path = os.path.join(args.input_dir, image_file)
        image = Image.open(image_path).convert('RGB')
        
        # Estimate depth
        depth = depth_estimator.estimate([image])[0]
        
        # Save depth map
        output_path = os.path.join(args.output_dir, f"{image_file}")
        depth.save(output_path)

    progress_bar.close()
    print("All images processed.")

if __name__ == "__main__":
    args = parse_args()
    main(args)