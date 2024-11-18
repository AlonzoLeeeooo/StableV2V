import os
import cv2
import numpy as np
from tqdm import tqdm

def dilate_mask(mask, kernel_size=11, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask

def process_masks(input_folder, output_folder, kernel_size=5, iterations=1):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all files in the input folder
    if input_folder.endswith('.png') or input_folder.endswith('.jpg'):
        mask_files = [input_folder]
    else:
        mask_files = [f for f in os.listdir(input_folder) if f.endswith('.png') or f.endswith('.jpg')]

    for mask_file in tqdm(mask_files, desc="Processing masks"):
        # Read the mask
        mask_path = os.path.join(input_folder, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Dilate the mask
        dilated_mask = dilate_mask(mask, kernel_size, iterations)
        

        # Save the dilated mask
        output_path = os.path.join(output_folder, mask_file)
        cv2.imwrite(output_path, dilated_mask)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dilate masks and save them in a new folder")
    parser.add_argument("--input-folder", type=str, required=True, help="Path to the folder containing input masks")
    parser.add_argument("--output-folder", type=str, required=True, help="Path to the folder to save dilated masks")
    parser.add_argument("--kernel-size", type=int, default=15, help="Kernel size for dilation (default: 5)")
    parser.add_argument("--iterations", type=int, default=1, help="Number of dilation iterations (default: 1)")

    args = parser.parse_args()

    process_masks(args.input_folder, args.output_folder, args.kernel_size, args.iterations)
    print(f"Dilated masks saved in {args.output_folder}.")
