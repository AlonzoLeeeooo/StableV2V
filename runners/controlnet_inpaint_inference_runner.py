import os
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, UniPCMultistepScheduler
from models.controlnet_inpaint.pipeline import StableDiffusionControlNetInpaintPipeline

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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
         args.stable_diffusion_inpaint_checkpoint_path,
         controlnet=controlnet,
         torch_dtype=weight_dtype,
         safety_checker=None,
     )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    # Load in everything
    image = Image.open(os.path.join(args.source_video_frames, sorted(os.listdir(args.source_video_frames))[0]))
    mask = Image.open(os.path.join(args.input_masks, sorted(os.listdir(args.input_masks))[0]))
    image = image.resize((args.width, args.height))
    mask = mask.resize((args.width, args.height))
    
    # Dilate the mask
    mask_np = np.array(mask)
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
    dilated_mask_np = cv2.dilate(mask_np, kernel, iterations=args.dilation_iterations)
    dilated_mask = Image.fromarray(dilated_mask_np)

    # Load MiDaS model for depth estimation
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()

    # MiDaS transformation
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform

    # Prepare image for MiDaS
    img = np.array(image)  # Convert PIL Image to numpy array
    img = transform(img).to(device)

    # Estimate depth
    with torch.no_grad():
        depth = midas(img)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(args.height, args.width),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Normalize depth map
    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = (depth * 255).astype(np.uint8)
    # Save depth map locally
    depth_save_path = os.path.join(args.outdir, 'depth_maps')
    os.makedirs(depth_save_path, exist_ok=True)
    depth_image = Image.fromarray(depth)
    depth_filename = f'depth_{os.path.splitext(os.path.basename(args.source_video_frames))[0]}.png'
    depth_image.save(os.path.join(depth_save_path, depth_filename))

    # Create control image from depth map
    control_image = Image.fromarray(depth)
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