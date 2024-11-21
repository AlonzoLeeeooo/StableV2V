import cv2
import os
import PIL
import torch
import numpy as np
from diffusers import PaintByExamplePipeline

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def paint_by_example_inference_runner(args):
    model_path = args.paint_by_example_checkpoint_path
    reference_image_path = args.external_guidance
    outdir = args.outdir
    height = args.height
    width = args.width

    # Create output directory if not existed
    os.makedirs(outdir, exist_ok=True)

    # Prepare inputs
    image_path = sorted(os.listdir(args.source_video_frames))[0]
    mask_path = sorted(os.listdir(args.input_masks))[0]
    init_image = PIL.Image.open(os.path.join(args.source_video_frames, image_path)).resize((height, width))
    mask_image = PIL.Image.open(os.path.join(args.input_masks, mask_path)).resize((height, width))
    reference_image = PIL.Image.open(reference_image_path).resize((height, width))


    # Dilate the mask to ensure that it covers the original object
    mask_np = np.array(mask_image)
    kernel = np.ones((args.kernel_size, args.kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask_np, kernel, iterations=args.dilation_iteration)
    mask_image = PIL.Image.fromarray(dilated_mask)



    # Prepare pipeline
    torch_dtype = torch.float32
    if args.mixed_precision == "fp32":
        torch_dtype = torch.float32
    elif args.mixed_precision == "fp16":
        torch_dtype = torch.float16
    pipe = PaintByExamplePipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(device)

    # Send inputs into the pipeline
    image = pipe(image=init_image,
                 mask_image=mask_image,
                 example_image=reference_image,
                 guidance_scale=args.guidance_scale,
                 negative_prompt=args.negative_prompt).images[0]

    # Save image
    save_path = os.path.join(outdir, 'image_editing_results')
    os.makedirs(save_path, exist_ok=True)
    filename = args.prompt.lower().replace('.', '').replace(' ', '_')
    image.save(os.path.join(save_path, f'{filename}.png'))
