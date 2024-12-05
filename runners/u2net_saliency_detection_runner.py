import os
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
from tqdm import tqdm
from models.u2net.u2net import U2NET

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(filename, pred, outdir, height, width):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    imo = im.resize((height, width),resample=Image.BILINEAR)

    imo.save(os.path.join(outdir, filename.split('.')[0] + '.png'))

def u2net_saliency_detection_runner(args, edited_first_frame_input_dir=None):

    # 1. Define basic configurations
    if edited_first_frame_input_dir is None:
        input_dir = args.source_video_frames
    else:
        input_dir = edited_first_frame_input_dir
    if edited_first_frame_input_dir is None:
        prediction_dir = os.path.join(args.outdir, 'iterative_warping', 'object_masks')
    else:
        prediction_dir = os.path.join(args.outdir, 'iterative_warping', 'editing_masks')
    model_dir = args.u2net_checkpoint_path


    # 2. Define model
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    

    # 3. Inference
    count = 0
    for frame_path in sorted(os.listdir(args.source_video_frames))[:args.n_sample_frames]:
        count += 1
        # Set a loop breaker if there is an edited first frame
        if args.edited_first_frame is not None: 
            if count >= 2:
                break
        
        # 3.1 Load in video frames
        if args.edited_first_frame is not None:
            frame = Image.open(args.edited_first_frame)
        else:   
            frame = Image.open(os.path.join(input_dir, frame_path))
        frame = frame.convert('RGB')
        frame = np.array(frame)

        # 3.2 Pass the first frame forward the model
        # The same as the original implementation of U2-Net
        inputs_test = frame
        inputs_test = torch.from_numpy(inputs_test.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # 3.3 Save outputted mask
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        if edited_first_frame_input_dir is None:
            save_output(frame_path, pred, prediction_dir, args.height, args.width)
        else:
            save_output(args.prompt.lower().replace(' ', '_'), pred, prediction_dir, args.height, args.width)

        del d1,d2,d3,d4,d5,d6,d7


def u2net_saliency_detection_for_single_image(args, edited_first_frame_input_dir=None):

    # 1. Define basic configurations
    input_dir = edited_first_frame_input_dir
    prediction_dir = os.path.join(args.outdir, 'reference_masks')
    model_dir = args.u2net_checkpoint_path


    # 2. Define model
    net = U2NET(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    

    # 3. Inference
    count = 0
    for frame_path in tqdm(sorted(os.listdir(args.source_video_frames))[:args.n_sample_frames]):
        count += 1
        # Set a loop breaker if there is an edited first frame
        if args.reference_image is not None: 
            if count >= 2:
                break
        
        # 3.1 Load in video frames
        frame = Image.open(args.reference_image)
        frame = frame.convert('RGB')
        frame = np.array(frame)

        # 3.2 Pass the first frame forward the model
        # The same as the original implementation of U2-Net
        inputs_test = frame
        inputs_test = torch.from_numpy(inputs_test.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        # 3.3 Save outputted mask
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(args.prompt.lower().replace(' ', '_'), pred, prediction_dir, args.height, args.width)

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-video-frames", type=str, default='assets/evaluation/images', required=True, help="Input directory of source video frames")
    parser.add_argument("--outdir", type=str, default='outputs/results', required=True, help="The output folder path to save generated videos")
    parser.add_argument("--u2net-checkpoint-path", type=str, default='', help='U2-Net checkpoint path')
    parser.add_argument("--n-sample-frames", type=int, default=16, help="Number of frames to sample")
    parser.add_argument("--height", type=int, default=512, help="Height of the video frames")
    parser.add_argument("--width", type=int, default=512, help="Width of the video frames")
    args = parser.parse_args()
    u2net_saliency_detection_runner(args)