import os
from PIL import Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


def instructpix2pix_inference_runner(args):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Weight dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Define pipeline
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.instructpix2pix_checkpoint_path,
        torch_dtype=weight_dtype,
        safety_checker=None)
    pipeline.to(device)
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)

    # Load the first-frame input
    input_list = sorted(os.listdir(args.source_video_frames))
    image = Image.open(os.path.join(args.source_video_frames, input_list[0]))
    image = image.resize((args.height, args.width))

    # Forward
    images = pipeline(args.external_guidance,
                      image=image,
                      seed=args.seed,
                      guidance_scale=args.guidance_scale,
                      negative_prompt=args.negative_prompt,
                      num_inference_steps=args.num_inference_steps,
                      image_guidance_scale=args.image_guidance_scale).images[0]
    
    # Save image
    save_path = os.path.join(args.outdir, 'image_editing_results')
    os.makedirs(save_path, exist_ok=True)
    filename = args.prompt.lower().replace('.', '').replace(' ', '_')
    images.save(os.path.join(save_path, f'{filename}.png'))