import os
from PIL import Image

import torch
from models.ctrl_adapter.controlnet import ControlNetModel
from models.ctrl_adapter.ctrl_adapter import ControlNetAdapter
from models.i2vgenxl.i2vgenxl_unet import I2VGenXLUNet
from models.i2vgenxl.i2vgenxl_ctrl_adapter_pipeline import I2VGenXLControlNetAdapterPipeline
from transformers import AutoTokenizer, CLIPTextModel
from utils.utils import save_as_gif


def i2vgenxl_ctrl_adapter_inference_runner(args):
    filename = args.prompt.replace('.', '').replace(' ', '_')
    # 1. Create output folder
    output_dir = os.path.join(args.outdir, 'generator_outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 2. Create folder of input frames
    input_frames_dir = os.path.join(output_dir, f"input_frames")
    os.makedirs(input_frames_dir, exist_ok=True)
 
    # 3. Create folder of output frames
    output_frames_dir = os.path.join(output_dir, f"output_frames")
    os.makedirs(output_frames_dir, exist_ok=True)

    # 4. Create folder of output frames of conditions
    output_condition_frames_dir = os.path.join(output_dir, f"conditon_frames")
    os.makedirs(output_condition_frames_dir, exist_ok=True)
    
    # 5. Create folder of input gifs
    input_gifs_dir = os.path.join(output_dir, "input_gifs")
    os.makedirs(input_gifs_dir, exist_ok=True)
    
    # 6. Create folder of output gifs 
    output_gifs_dir = os.path.join(output_dir, "output_gifs")
    os.makedirs(output_gifs_dir, exist_ok=True)
    
    # 7. Create folder of condition gifs
    output_condition_gifs_dir = os.path.join(output_dir, f"condition_gifs")
    os.makedirs(output_condition_gifs_dir, exist_ok=True)
    
    # 8. (Optional) Create folder of the first frame used in each generation round
    output_image_input_dir = os.path.join(output_dir, f"sanity_check_image_inputs")
    os.makedirs(output_image_input_dir, exist_ok=True)
    

    # Define device and inference precision
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    data_type = torch.float32
    if args.mixed_precision == 'f16':
        data_type = torch.half
    elif args.mixed_precision == 'f32':
        data_type = torch.float32
    elif args.mixed_precision == 'bf16':
        data_type = torch.bfloat16


    # Load adapter 
    adapter = ControlNetAdapter.from_pretrained(
        args.ctrl_adapter_checkpoint_path,
        low_cpu_mem_usage=False, 
        device_map=None
        )
    adapter = adapter.to(data_type)
    adapter.eval()


    # Initialize configurations for ControlNet
    pipeline_args = {
        "torch_dtype": data_type, 
        "use_safetensors": True, 
        'adapter': adapter
        }
    pipeline_args['controlnet'] = {}
    
    # Load ControlNet
    pipeline_args['controlnet'] = ControlNetModel.from_pretrained(
        args.controlnet_checkpoint_path,
        torch_dtype=data_type,
        use_safetensors=True)
    pipeline_args['controlnet_text_encoder'] = CLIPTextModel.from_pretrained(
        args.stable_diffusion_checkpoint_path,
        subfolder="text_encoder",
        torch_dtype=data_type,
        use_safetensors=True).to(device, dtype=data_type)
    pipeline_args['controlnet_tokenizer'] = AutoTokenizer.from_pretrained(
        args.stable_diffusion_checkpoint_path,
        subfolder="tokenizer",
        use_fast=False
    )

        
    # Load video generation model
    pipe = I2VGenXLControlNetAdapterPipeline.from_pretrained(
        args.i2vgenxl_checkpoint_path,
        **pipeline_args).to(device, dtype=data_type)
    pipe.unet = I2VGenXLUNet.from_pretrained(args.i2vgenxl_checkpoint_path, subfolder="unet").to(device, dtype=data_type)
    

    # Initialize random seed
    generator = torch.Generator().manual_seed(args.seed) if args.seed else None

    # Start generation
    print(f"Prompt: {args.prompt}.")
    
    
    # NOTE: Load input video frames
    # 1. Load in the paths
    input_frame_paths = sorted(os.listdir(args.source_video_frames))
    
    # 2. Pre-process the paths
    filtered_input_frame_paths = []
    for input_frame_path in input_frame_paths:
        if 'jpg' in input_frame_path or 'png' in input_frame_path:
            filtered_input_frame_paths.append(input_frame_path)
    filtered_input_frame_paths = sorted(filtered_input_frame_paths)[:args.n_sample_frames]
    
    # 3. Load in frames in the filtered paths
    input_frames = []
    for input_frame_path in filtered_input_frame_paths:
        input_frame = Image.open(os.path.join(args.source_video_frames, input_frame_path))
        input_frames.append(input_frame)
        
    # 4. Resize the input frames
    processed_input_frames = []
    for input_frame in input_frames:
        processed_input_frame = input_frame.resize((args.width, args.height))
        processed_input_frames.append(processed_input_frame)
        
        
    # 5. Double-check that only the first `n_sample_frames` frames are used
    final_input_frames = processed_input_frames[:args.n_sample_frames]
    

    # 6. Load input conditions
    input_condition_path = os.path.join(args.outdir, 'iterative_warping', 'final_depth_maps')
    input_condition_paths = sorted(os.listdir(input_condition_path))[:args.n_sample_frames]
    input_conditions = []
    for path in input_condition_paths:
        input_condition = Image.open(os.path.join(input_condition_path, path))
        input_condition = input_condition.resize((args.width, args.height))
        input_conditions.append(input_condition)
    control_images = input_conditions

    
    # Load in edited first frame
    if args.edited_first_frame is not None:
        edited_first_frame = Image.open(args.edited_first_frame)
    
    
    # Intiailize inference configurations
    kwargs = {
        'controlnet_conditioning_scale': args.controlnet_conditioning_scale,
        'control_guidance_start': args.control_guidance_start,
        'control_guidance_end': args.control_guidance_end,

        'sparse_frames': args.sparse_frames,
        'skip_conv_in': args.skip_conv_in,
        'skip_time_emb': False,
        'use_size_512': True,
    }
    
        
    # NOTE: Initialize image input
    image_input = edited_first_frame
        
        
    # Execute with GPU
    if device == 'cuda':
        with torch.autocast(device_type=device, dtype=data_type):
            output_frames = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height, 
                width=args.width,
                image=image_input,
                control_images=control_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, 
                generator=generator,
                target_fps=args.output_fps,
                num_frames=args.n_sample_frames,
                output_type="pil",
                **kwargs
            ).frames[0]
    # Execute with other devices, e.g., Mac Silicon or CPU
    else:
        output_frames = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height, 
                width=args.width,
                image=image_input,
                control_images=control_images,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, 
                generator=generator,
                target_fps=args.output_fps,
                num_frames=args.n_sample_frames,
                output_type="pil",
                **kwargs
            ).frames[0]
            

    # NOTE: Save everything
    # 1. Save input frames
    for i in range(len(final_input_frames))[:args.n_sample_frames]:
        final_input_frames[i].save(os.path.join(input_frames_dir, f"{i:05d}.png"))

    # 2. Save input gifs
    save_as_gif(final_input_frames[:args.n_sample_frames], os.path.join(input_gifs_dir, f"{filename}.gif"), duration=args.video_duration // args.video_length)
    
    # 3. Save condition frames
    for i in range(len(final_input_frames))[:args.n_sample_frames]:
        control_images[i].save(os.path.join(output_condition_frames_dir, f"{i:05d}.png"))
    
    # 4. Save condition gif
    save_as_gif(control_images, os.path.join(output_condition_gifs_dir, f"{filename}.gif"), duration=args.video_duration // args.video_length)
    
    # 5. Save output frames
    for i in range(len(output_frames))[:args.n_sample_frames]:
        output_frames[i].save(os.path.join(output_frames_dir, f"{i:05d}.png"))
    
    # 6. Save output gif, this will iteratively go on as the generation proceeds
    save_as_gif(output_frames, os.path.join(output_gifs_dir, f"{filename}.gif"), duration=args.video_duration // args.video_length)



