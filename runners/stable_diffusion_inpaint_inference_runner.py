import torch
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline



def stable_diffusion_inpaint_inference_runner(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Define weight dtype
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.stable_diffusion_inpaint_checkpoint_path,
        torch_dtype=weight_dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device, dtype=weight_dtype)

    # Load the first frame from the source video frames
    image = Image.open(os.path.join(args.source_video_frames, sorted(os.listdir(args.source_video_frames))[0])).convert("RGB")
    # Load the first mask from the input masks
    mask = Image.open(os.path.join(args.input_masks, sorted(os.listdir(args.input_masks))[0])).convert("RGB")

    # Convert mask to numpy array
    mask_np = np.array(mask)

    # Create a kernel for dilation
    kernel = np.ones((19, 19), np.uint8)

    # Dilate the mask
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=9)

    # Convert back to PIL Image
    mask_image = Image.fromarray(dilated_mask)

    generator = torch.Generator().manual_seed(args.seed)

    output_image = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=mask_image,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        generator=generator
    ).images[0]

    # Save image
    save_path = os.path.join(args.outdir, 'image_editing_results')
    os.makedirs(save_path, exist_ok=True)
    filename = args.prompt.replace('.', '').replace(' ', '_')
    output_image.save(os.path.join(save_path, f'{filename}.png'))