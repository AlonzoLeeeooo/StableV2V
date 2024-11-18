import argparse
import os
import shutil
import time
from runners.paint_by_example_inference_runner import paint_by_example_inference_runner
from runners.instructpix2pix_inference_runner import instructpix2pix_inference_runner
from runners.controlnet_inpaint_inference_runner import controlnet_inpaint_inference_runner
from runners.completion_net_inference_runner import depth_completion_net_inference_runner
from runners.i2vgenxl_ctrl_adapter_inference_runner import i2vgenxl_ctrl_adapter_inference_runner
from runners.iterative_warping.run_flow_extraction import raft_flow_extraction_runner
from runners.midas_depth_estimation_runner import midas_depth_estimation_runner
from runners.u2net_saliency_detection_runner import u2net_saliency_detection_runner
from runners.iterative_warping_runner import iterative_warping_runner
from runners.stable_diffusion_inpaint_inference_runner import stable_diffusion_inpaint_inference_runner
from runners.anydoor_inference_runner import anydoor_inference_runner
from utils.utils import bool_flag

DEFAULT_NEGATIVE_PROMPTS = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"

def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    # Model configurations
    # Image editors
    parser.add_argument("--image-editor-checkpoint-path", type=str, default='', help="Image editor checkpoint path")
    parser.add_argument("--paint-by-example-checkpoint-path", type=str, default='', help="Paint-by-Example checkpoint path")
    parser.add_argument("--instructpix2pix-checkpoint-path", type=str, default='', help="InstructPix2Pix checkpoint path")
    parser.add_argument("--controlnet-inpaint-checkpoint-path", type=str, default='', help="ControlNet Inpaint checkpoint path")
    parser.add_argument("--stable-diffusion-inpaint-checkpoint-path", type=str, default='', help="Stable Diffusion Inpaint checkpoint path")
    parser.add_argument("--anydoor-checkpoint-path", type=str, default='', help='AnyDoor checkpoint path')
    parser.add_argument("--anydoor-config-path", type=str, default='models/anydoor/configs/inference.yaml', help='AnyDoor config path')
    # Iterative warping
    parser.add_argument("--raft-checkpoint-path", type=str, default='', help='RAFT checkpoint path')
    parser.add_argument("--midas-checkpoint-path", type=str, default='', help='MiDaS checkpoint path')
    parser.add_argument("--u2net-checkpoint-path", type=str, default='', help='U2-Net checkpoint path')
    # Image-to-video generators
    parser.add_argument("--stable-diffusion-checkpoint-path", type=str, default='', help='Huggingface repo of Stable Diffusion v1.5')
    parser.add_argument("--controlnet-checkpoint-path", type=str, default='', help='Huggingface repo of ControlNet v1.1')
    parser.add_argument("--i2vgenxl-checkpoint-path", type=str, default='', help='Huggingface repo of I2VGenXL')
    parser.add_argument("--ctrl-adapter-checkpoint-path", type=str, default='', help='Huggingface repo of Ctrl Adapter')
    parser.add_argument("--completion-net-checkpoint-path", type=str, default='', help='Huggingface repo of Completion Net')


    parser.add_argument("--xformers", action="store_true")
    parser.add_argument('--sparse-frames', nargs='+', default=None, help="Original sparse frames implementation of Ctrl Adapter")
    parser.add_argument('--skip-conv-in', default=False, type=bool_flag, help="Latents skipping strategy in Ctrl Adapter")
    
    # Inference configurations
    # Input arguments
    parser.add_argument("--source-video-frames", type=str, default='', required=True, help="Input directory of source video frames")
    parser.add_argument("--input-masks", type=str, default=None, help="Input masks")
    parser.add_argument('--edited-first-frame', type=str, default=None, help='Path of the edited first frame')
    parser.add_argument("--prompt", type=str, default='', required=True, help='Text prompt')
    
    parser.add_argument("--input-condition", type=str, default=None, help="Directly load conditions from local path is this variable is not None, otherwise extract conditions from source video frames")
    parser.add_argument("--external-guidance", type=str, default=None, help="External for image editor")
    parser.add_argument("--reference-masks", type=str, default='', help="Masks of reference image for AnyDoor")
    parser.add_argument("--image-guidance-scale", type=float, default=1.0, help="Image guidance scale for InstructPix2Pix")
    parser.add_argument("--kernel-size", type=int, default=9, help="Kernel size for dilation")
    parser.add_argument("--dilation-iteration", type=int, default=1, help="Dilation iteration")
    parser.add_argument("--outdir", type=str, default='outputs/results', required=True, help="The output folder path to save generated videos")

    # Image editor arguments
    parser.add_argument("--image-editor-type", type=str, default='paint-by-example', choices=['paint-by-example', 'instructpix2pix', 'controlnet-inpaint', 'stable-diffusion-inpaint', 'anydoor'], help="Image editor type")

    # Image-to-video generator arguments
    parser.add_argument("--image-to-video-generator-type", type=str, default='i2vgenxl', choices=['i2vgenxl'], help="Image-to-video generator type")

    # Other arguments
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPTS, help='Negative prompt')
    parser.add_argument("--guidance-scale", type=float, default=9.0, help='Scale of classifier-free guidance')
    parser.add_argument("--n-sample-frames", type=int, default=16, help="Number of output frames of video generation model")
    parser.add_argument("--mixed-precision",type=str, default='bf16', choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--width", type=int, default=512, help="Width of generated video frames")
    parser.add_argument("--height", type=int, default=512, help="Height of generated video frames")
    parser.add_argument("--video-length", type=int, default=8, help="Video length of saving output gif")
    parser.add_argument("--video-duration", type=int, default=1000, help="Video duration of saving output gif")
    parser.add_argument("--output-fps", type=int, default=16, help='Output FPS')
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # ControlNet configurations
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=1.0, help="Conditioning scale of ControlNet")
    parser.add_argument("--control-guidance-start", type=float, default=0.0, help="Where the control guidance starts")
    parser.add_argument("--control-guidance-end", type=float, default=1.0, help="Where the control ends")
    return parser



if __name__ == "__main__":
    # Define the configurations
    parser = argparse.ArgumentParser(parents=[parse_args()])
    args = parser.parse_args()

    if args.input_masks is None:
        u2net_saliency_detection_runner(args)
        args.input_masks = os.path.join(args.outdir, 'iterative_warping', 'object_masks')

    # 1. Run image editing on the first video frame
    if args.edited_first_frame is None:
        if args.image_editor_type == 'paint-by-example':
            args.paint_by_example_checkpoint_path = args.image_editor_checkpoint_path
            assert args.external_guidance is not None, "External guidance must be provided for `Paint-by-Example` editor."
            paint_by_example_inference_runner(args)
        elif args.image_editor_type == "instructpix2pix":
            args.instructpix2pix_checkpoint_path = args.image_editor_checkpoint_path
            assert args.external_guidance is not None, "External guidance must be provided for `InstructPix2Pix` editor."
            instructpix2pix_inference_runner(args)
        elif args.image_editor_type == "controlnet-inpaint":
            args.stable_diffusion_inpaint_checkpoint_path = args.image_editor_checkpoint_path
            assert args.external_guidance is not None, "External guidance must be provided for `ControlNet Inpaint` editor."
            controlnet_inpaint_inference_runner(args)
        elif args.image_editor_type == "stable-diffusion-inpaint":
            args.stable_diffusion_inpaint_checkpoint_path = args.image_editor_checkpoint_path
            stable_diffusion_inpaint_inference_runner(args)
        elif args.image_editor_type == "anydoor":
            args.anydoor_checkpoint_path = args.image_editor_checkpoint_path
            anydoor_inference_runner(args)
    print("\n1. Image editing done.\n")

    # 2. Run iterative warping
    # Extract optical flows from source video frames
    raft_flow_extraction_runner(args)
    print("\n2. Optical flow extraction done.\n")

    # Extract depth maps from source video frames
    midas_depth_estimation_runner(args)
    print("\n3. Depth map extraction done.\n")

    # Extract shape mask from the edited first frame
    # Extract object masks from the source video frames
    if args.input_masks is None:
        u2net_saliency_detection_runner(args)
    else:
        # Copy files in args.input_masks to the object_masks directory
        object_masks_dir = os.path.join(args.outdir, 'iterative_warping', 'object_masks')
        os.makedirs(object_masks_dir, exist_ok=True)
        for mask_file in sorted(os.listdir(args.input_masks))[:args.n_sample_frames]:
            src_path = os.path.join(args.input_masks, mask_file)
            dst_path = os.path.join(object_masks_dir, mask_file)
            if os.path.isfile(src_path) and not os.path.isfile(dst_path):
                shutil.copy(src_path, dst_path)
    # Extract editing masks from the edited first frame
    if args.edited_first_frame is None:
        args.edited_first_frame = os.path.join(args.outdir, 'image_editing_results', f'{args.prompt.lower().replace(" ", "_")}.png')
    u2net_saliency_detection_runner(args, args.edited_first_frame)
    print("\n4. Object and editing masks extraction done.\n")

    # Get edited optical flows and depth maps
    iterative_warping_runner(args)
    print("\n5. Iterative warping done.\n")

    # Run depth completion net to remove extra region
    depth_completion_net_inference_runner(args)
    print("\n6. Depth completion done.\n")

    # 3. Run image-to-video generation with the edited depth maps
    i2vgenxl_ctrl_adapter_inference_runner(args)
    print("\n7. Image-to-video generation done.\n")
