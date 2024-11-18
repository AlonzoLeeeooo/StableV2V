import torch
import argparse

DEFAULT_NEGATIVE_PROMPTS = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"

import os
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from models.controlnet_inpaint.pipeline import StableDiffusionControlNetInpaintPipeline

def controlnet_inpaint_inference_runner(args):
    weight_dtype = torch.float32
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16

    # Define ControlNet and Stable Diffusion
    controlnet = ControlNetModel.from_pretrained(args.controlnet_checkpoint_path,
                                                 torch_dtype=weight_dtype)
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
         args.stable_diffusion_checkpoint_path,
         controlnet=controlnet,
         torch_dtype=weight_dtype,
         safety_checker=None,
     )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(args.device)

    # Load in everything
    image = Image.open(os.path.join(args.source_video_frames, sorted(os.listdir(args.source_video_frames))[0]))
    mask = Image.open(os.path.join(args.input_mask))
    image = image.resize((args.width, args.height))
    mask = mask.resize((args.width, args.height))
    
    # Dilate the mask
    mask_np = np.array(mask)
    mask_np[mask_np > 127.5] = 255
    mask_np[mask_np <= 127.5] = 0
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
    dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=args.dilate_iterations)
    dilated_mask = Image.fromarray(dilated_mask_np)

    # Create control image from depth map
    if args.controlnet_guidance is None:
        # Extract depth using MiDaS
        from transformers import pipeline
        depth_estimator = pipeline('depth-estimation', model='Intel/dpt-hybrid-midas')
        depth_map = depth_estimator(image)['depth']
        control_image = depth_map
    else:
        control_image = Image.open(args.controlnet_guidance)
        control_image = control_image.resize((args.width, args.height))

    # Generate image
    generator = torch.manual_seed(args.seed)
    new_image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        image=image,
        control_image=control_image,
        mask_image=dilated_mask,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt,
    ).images[0]

    
    
    # Save image
    save_path = os.path.join(args.outdir, 'image_editing_results')
    os.makedirs(save_path, exist_ok=True)
    
    filename = args.prompt.replace('.', '').replace(' ', '_')
    new_image.save(os.path.join(save_path, f'{filename}.png'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
                        help="Device to run the model on (cuda, mps, or cpu)")
    parser.add_argument("--controlnet-checkpoint-path", type=str, 
                        default='',
                        help="Path to the ControlNet checkpoint")
    parser.add_argument("--stable-diffusion-checkpoint-path", type=str, 
                        default='',
                        help="Path to the Stable Diffusion checkpoint")
    parser.add_argument("--prompt", type=str, default="a yellow duck swimming in the river",
                        help="Prompt for image generation")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPTS, help="Negative prompt")
    parser.add_argument("--source-video-frames", type=str, default='',
                        help="Path to the input image")
    parser.add_argument("--input-mask", type=str, default='',
                        help="Path to the mask image")
    parser.add_argument("--controlnet-guidance", type=str, default=None,
                        help="Path to the condition image")
    parser.add_argument("--height", type=int, default=512, help="Height of the output image")
    parser.add_argument("--width", type=int, default=512, help="Width of the output image")
    parser.add_argument("--num-inference-steps", type=int, default=20,
                        help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=23, help="Random seed for generation")
    parser.add_argument("--outdir", type=str, default='controlnet_inpaint_result.png',
                        help="Path to save the output image")
    parser.add_argument("--mixed-precision", type=str, default='no', choices=['no', 'fp16', 'bf16'], help="Argument to decide the weight dtype")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--kernel-size", type=int, default=19, help="Kernel size for dilation")
    parser.add_argument("--dilate-iterations", type=int, default=5, help="Number of dilation iterations")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    controlnet_inpaint_inference_runner(args)