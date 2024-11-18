import argparse
import cv2
import os

def get_editing_region(src_mask, approximate_mask):
    src_mask = src_mask / 255.
    approximate_mask = approximate_mask / 255.
    editing_region = src_mask * (1 - approximate_mask)
    return (editing_region * 255.).astype('uint8')

def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of all files in the source mask directory
    src_mask_files = sorted([f for f in os.listdir(args.src_mask_dir) if f.endswith('.png')])
    approx_mask_files = sorted([f for f in os.listdir(args.approx_mask_dir) if f.endswith('.png')])

    for count, (src_filename, approx_filename) in enumerate(zip(src_mask_files, approx_mask_files)):
        # Load source mask
        src_mask_path = os.path.join(args.src_mask_dir, src_filename)
        src_mask = cv2.imread(src_mask_path)
        
        # Load approximate mask
        approx_mask_path = os.path.join(args.approx_mask_dir, approx_filename)
        approximate_mask = cv2.imread(approx_mask_path)

        # Resize masks to the same size
        if src_mask.shape != approximate_mask.shape:
            height, width = src_mask.shape[:2]
            approximate_mask = cv2.resize(approximate_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Get editing region
        editing_region = get_editing_region(src_mask, approximate_mask)

        # Save editing region mask
        output_path = os.path.join(args.output_dir, f'{count:05d}.png')
        cv2.imwrite(output_path, editing_region)

        print(f"Progress: {count + 1}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate editing region masks for video frames")
    parser.add_argument("--src-mask-dir", type=str, required=True, help="Directory containing source mask frames")
    parser.add_argument("--approx-mask-dir", type=str, required=True, help="Directory containing approximate mask frames")
    parser.add_argument("--output-dir", type=str, default="editing_regions", help="Output directory for editing region masks")
    
    args = parser.parse_args()
    main(args)