import os
import cv2
import tempfile
import shutil
from PIL import Image
import gradio as gr

from inference import parse_args
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

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Initialize default arguments
args = parse_args()

# TODO: Model paths configuration
args.paint_by_example_checkpoint_path = "checkpoints/paint-by-example"
args.instructpix2pix_checkpoint_path = "checkpoints/instruct-pix2pix"
args.stable_diffusion_inpaint_checkpoint_path = "checkpoints/stable-diffusion-inpaint"
args.anydoor_checkpoint_path = "checkpoints/anydoor"
args.raft_checkpoint_path = "checkpoints/raft/raft-things.pth"
args.midas_checkpoint_path = "checkpoints/dpt_swin2_large_384.pt"
args.u2net_checkpoint_path = "checkpoints/u2net.pth"
args.stable_diffusion_checkpoint_path = "checkpoints/stable-diffusion-v1.5"
args.controlnet_checkpoint_path = "checkpoints/controlnet-depth"
args.ctrl_adapter_checkpoint_path = "checkpoints/ctrl-adapter-i2vgenxl-depth"
args.i2vgenxl_checkpoint_path = "checkpoints/i2vgenxl"
args.completion_net_checkpoint_path = "checkpoints/50000.ckpt"
args.edited_first_frame = None


def run_stablev2v(
    video_input,
    prompt,
    image_editor_type,
    external_guidance_text=None,
    external_guidance_image=None,
    negative_prompt=None,
    guidance_scale=9.0,
    n_frames=16,
    output_fps=16,
    height=512,
    width=512,
    edited_first_frame=None,
    kernel_size=11,
    dilation_iteration=3,
    mixed_precision='bf16',
    seed=42,
):
    # Create temporary directories for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        

        # Extract frames from video to temp directory
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        
        # Set up arguments
        args.prompt = prompt
        args.image_editor_type = image_editor_type
        args.height = height
        args.width = width
        args.n_sample_frames = n_frames
        args.output_fps = output_fps
        args.guidance_scale = guidance_scale
        args.outdir = "results"
        args.kernel_size = kernel_size
        args.dilation_iteration = dilation_iteration
        args.mixed_precision = mixed_precision
        args.seed = seed
        args.controlnet_conditioning_scale = 1.0
        args.control_guidance_start = 0.0
        args.control_guidance_end = 1.0
        args.sparse_frames = None
        args.skip_conv_in = False
        args.num_inference_steps = 50
        args.video_length = 8
        args.video_duration = 1000
        args.input_condition = None
        args.reference_image = None
        args.reference_masks = None
        args.image_guidance_scale = 1.0
        args.anydoor_config_path = 'models/anydoor/configs/inference.yaml'

        os.makedirs(args.outdir, exist_ok=True)
        os.makedirs(os.path.join(args.outdir, 'frames'), exist_ok=True)
        if negative_prompt:
            args.negative_prompt = negative_prompt
        for video_frame_path in video_input:
            current_frame = Image.open(video_frame_path)
            filename = os.path.basename(video_frame_path).split('.')[0] + '.png'
            frame_path = os.path.join(args.outdir, 'frames', filename)
            current_frame.save(frame_path)
        args.source_video_frames = os.path.join(args.outdir, 'frames')
        if image_editor_type in ['paint-by-example', 'anydoor'] and edited_first_frame is None:
            external_guidance_image.save(os.path.join(args.outdir, 'reference_image.png'))
            args.external_guidance = os.path.join(args.outdir, 'reference_image.png')
        elif image_editor_type == 'instructpix2pix' and edited_first_frame is None:
            args.external_guidance = external_guidance_text
        if edited_first_frame:
            edited_first_frame.save(os.path.join(args.outdir, 'edited_first_frame.png'))
            args.edited_first_frame = os.path.join(args.outdir, 'edited_first_frame.png') 
            
        # Run inference pipeline
        u2net_saliency_detection_runner(args)
        args.input_masks = os.path.join(args.outdir, 'iterative_warping', 'object_masks')

        # 1. Run image editing on the first video frame
        if args.edited_first_frame is None:
            if args.image_editor_type == 'paint-by-example':
                assert args.external_guidance is not None, "External guidance must be provided for `Paint-by-Example` editor."
                paint_by_example_inference_runner(args)
            elif args.image_editor_type == "instructpix2pix":
                assert args.external_guidance is not None, "External guidance must be provided for `InstructPix2Pix` editor."
                instructpix2pix_inference_runner(args)
            elif args.image_editor_type == "controlnet-inpaint":
                assert args.external_guidance is not None, "External guidance must be provided for `ControlNet Inpaint` editor."
                controlnet_inpaint_inference_runner(args)
            elif args.image_editor_type == "stable-diffusion-inpaint":
                stable_diffusion_inpaint_inference_runner(args)
            elif args.image_editor_type == "anydoor":
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

        def image_to_video(file_path, output, fps=24):
            img_list = sorted(os.listdir(file_path))
            image = cv2.imread(os.path.join(file_path, img_list[1]))
            height, width, _ = image.shape
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videowriter = cv2.VideoWriter(output, fourcc, fps, (width, height))
            for img in img_list:
                path = os.path.join(file_path, img)
                frame = cv2.imread(path)
                videowriter.write(frame)

            videowriter.release()
        
        # Return the generated gif path
        result_path = os.path.join(args.outdir, 'generator_outputs', 'output_frames')
        image_to_video(result_path, os.path.join(args.outdir, 'generator_outputs', 'output_video.mp4'), args.output_fps)
        return os.path.join(args.outdir, 'generator_outputs', 'output_video.mp4')


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Gradio Demo for StableV2V")            

    with gr.Row():
        # Left column - Input settings
        with gr.Column(scale=1):
            video_input = gr.Files(label="Input Frames Folder", file_count="directory")
            prompt = gr.Textbox(label="Prompt", value="a high quality video")
            image_editor_type = gr.Dropdown(
                label="Image Editor Type",
                choices=["paint-by-example", "instructpix2pix", "controlnet-inpaint", 
                        "stable-diffusion-inpaint", "anydoor"],
                value="paint-by-example"
            )
            submit_btn = gr.Button("Generate")

        # Middle column - Advanced settings
        with gr.Column(scale=1):
            # Guidance inputs
            with gr.Tabs():
                with gr.Tab("Reference Image"):
                    external_guidance_image = gr.Image(
                        label="Required by Paint-by-Example and AnyDoor", 
                        type='pil'
                    )
                with gr.Tab("User Instructions"):
                    external_guidance_text = gr.Textbox(
                        label="Required by InstructPix2Pix", 
                        type='text'
                    )
                with gr.Tab("First Edited Frame"):
                    edited_first_frame = gr.Image(
                        label="First Edited Frame", 
                        type='pil'
                    )

            with gr.Accordion("Advanced Options", open=False):
                mixed_precision = gr.Dropdown(
                    label="Mixed Precision",
                    choices=["fp32", "fp16", 'bf16'],
                    value="bf16",
                    type='value'
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
                )
                guidance_scale = gr.Slider(
                    minimum=1, maximum=20, value=9.0,
                    label="Guidance Scale"
                )
                n_frames = gr.Slider(
                    minimum=1, maximum=32, value=16, step=1,
                    label="Number of Frames"
                )
                output_fps = gr.Slider(
                    minimum=1, maximum=30, value=16, step=1,
                    label="Output FPS"
                )
                height = gr.Slider(
                    minimum=1, maximum=1024, value=512, step=1,
                    label="Height"
                )
                width = gr.Slider(
                    minimum=1, maximum=1024, value=512, step=1,
                    label="Width"
                )
                kernel_size = gr.Slider(
                    minimum=1, maximum=49, value=11, step=2,
                    label="Kernel Size"
                )
                dilation_iterations = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="Dilation Iterations"
                )
                seed = gr.Slider(
                    minimum=1, maximum=1000000, value=42, step=1,
                    label="Seed"
                )

        # Right column - Output
        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video")

    # Submit button click handler
    submit_btn.click(
        run_stablev2v,
        inputs=[
            video_input,
            prompt,
            image_editor_type,
            external_guidance_text,
            external_guidance_image,
            negative_prompt,
            guidance_scale,
            n_frames,
            output_fps,
            height,
            width,
            edited_first_frame,
            kernel_size,
            dilation_iterations,
            mixed_precision,
            seed,
        ],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch()
