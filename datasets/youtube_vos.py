"""
YouTube-VOS dataset for flow completion.
"""

import os
import json
import random

import cv2
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms

from utils.file_client import FileClient
from utils.flow_utils import resize_flow, flowread
from utils.mask_utils import create_random_shape_with_random_motion
from utils.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip, GroupRandomHorizontalFlowFlip, binarize_tensor

def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img

class YouTubeVOSFlowDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args.video_path
        self.flow_root = args.flow_path                             # TODO: Do we really need this?
        self.num_local_frames = args.num_local_frames
        self.num_ref_frames = args.num_ref_frames
        self.size = self.w, self.h = (args.width, args.height)

        self.load_flow = args.load_flow
        if self.load_flow is not None:
            assert os.path.exists(self.flow_root)
        
        json_path = args.annotation_path

        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict['videos'].keys()))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list
                

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        masks = []
        flows_f, flows_b = [], []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            frames.append(img)
            masks.append(all_masks[idx])

            if len(frames) <= self.num_local_frames-1 and self.load_flow is not None:
                current_n = frame_list[idx][:-4]
                next_n = frame_list[idx+1][:-4]
                flow_f_path = os.path.join(self.flow_root, video_name, f'{current_n}_{next_n}_f.flo')
                flow_b_path = os.path.join(self.flow_root, video_name, f'{next_n}_{current_n}_b.flo')
                flow_f = flowread(flow_f_path, quantize=False)
                flow_b = flowread(flow_b_path, quantize=False)
                flow_f = resize_flow(flow_f, self.h, self.w)
                flow_b = resize_flow(flow_b, self.h, self.w)
                flows_f.append(flow_f)
                flows_b.append(flow_b)

            if len(frames) == self.num_local_frames: # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    masks.reverse()
                    if self.load_flow is not None:
                        flows_f.reverse()
                        flows_b.reverse()
                        flows_ = flows_f
                        flows_f = flows_b
                        flows_b = flows_
                
        if self.load_flow is not None:
            frames, flows_f, flows_b = GroupRandomHorizontalFlowFlip()(frames, flows_f, flows_b)
        else:
            frames = GroupRandomHorizontalFlip()(frames)

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        if self.load_flow is not None:
            flows_f = np.stack(flows_f, axis=-1) # H W 2 T-1
            flows_b = np.stack(flows_b, axis=-1)
            flows_f = torch.from_numpy(flows_f).permute(3, 2, 0, 1).contiguous().float()
            flows_b = torch.from_numpy(flows_b).permute(3, 2, 0, 1).contiguous().float()

        # img [-1,1] mask [0,1]
        batch = {
            "frames": frame_tensors,
            "masks": mask_tensors,
            "forward_flow": flows_f if self.load_flow is not None else "None",
            "backward_flow": flows_b if self.load_flow is not None else "None",
            "video_name": video_name
        }
        return batch

class YouTubeVOSDepthDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args.video_path
        self.depth_path = args.depth_path
        self.shape_path = args.shape_path
        self.num_local_frames = args.num_local_frames
        self.num_ref_frames = args.num_ref_frames
        self.size = self.w, self.h = (args.width, args.height)
        json_path = args.annotation_path

        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict['videos'].keys()))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list
                

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
        # create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)

        # create sample index
        selected_index = self._sample_index(self.video_dict[video_name],
                                            self.num_local_frames,
                                            self.num_ref_frames)

        # read video frames
        frames = []
        masks = []
        depths = []
        shapes = []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            # Depth
            depth_path = os.path.join(self.depth_path, video_name, frame_list[idx].split('.')[0] + '.png')
            depth = Image.open(depth_path).convert('RGB').resize(self.size)

            # Shape
            shape_path = os.path.join(self.shape_path, video_name, frame_list[idx].split('.')[0] + '.png')
            shape = Image.open(shape_path).convert('L').resize(self.size)

            frames.append(img)
            masks.append(all_masks[idx])
            depths.append(depth)
            shapes.append(shape)

            if len(frames) == self.num_local_frames: # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    masks.reverse()
                    depths.reverse()
                    shapes.reverse()
                
        frames_pil = GroupRandomHorizontalFlip()(frames)
        depths = GroupRandomHorizontalFlip()(depths)
        shapes = GroupRandomHorizontalFlip()(shapes)

        # normalizate, to tensors
        frame_tensors = self._to_tensors(frames_pil) * 2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        depth_tensors = self._to_tensors(depths)
        shape_tensors = self._to_tensors(shapes)
        shape_tensors = binarize_tensor(shape_tensors)
        
        # img [-1,1] mask [0,1]
        batch = {
            "frames": frame_tensors,
            "masks": mask_tensors,
            "depths": depth_tensors,
            "shapes": shape_tensors,
        }
        return batch
    

class YouTubeVOSDeformationDataset(torch.utils.data.Dataset):
    def __init__(self, args: dict):
        self.args = args
        self.video_root = args.video_path
        self.depth_path = args.depth_path
        self.num_local_frames = args.num_local_frames
        self.num_ref_frames = args.num_ref_frames
        self.size = self.w, self.h = (args.width, args.height)
        json_path = args.annotation_path

        with open(json_path, 'r') as f:
            self.video_train_dict = json.load(f)
        self.video_names = sorted(list(self.video_train_dict['videos'].keys()))

        self.video_dict = {}
        self.frame_dict = {}

        for v in self.video_names:
            frame_list = sorted(os.listdir(os.path.join(self.video_root, v)))
            v_len = len(frame_list)
            if v_len > self.num_local_frames + self.num_ref_frames:
                self.video_dict[v] = v_len
                self.frame_dict[v] = frame_list
                

        self.video_names = list(self.video_dict.keys()) # update names

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(),
        ])
        self.file_client = FileClient('disk')

    def __len__(self):
        return len(self.video_names)

    def _sample_index(self, length, sample_length, num_ref_frame=3):
        complete_idx_set = list(range(length))
        pivot = random.randint(0, length - sample_length)
        local_idx = complete_idx_set[pivot:pivot + sample_length]
        remain_idx = list(set(complete_idx_set) - set(local_idx))
        ref_index = sorted(random.sample(remain_idx, num_ref_frame))

        return local_idx + ref_index

    def __getitem__(self, index):
        video_name = self.video_names[index]
            
        # Create masks
        all_masks = create_random_shape_with_random_motion(
            self.video_dict[video_name], imageHeight=self.h, imageWidth=self.w)


        # Create sample index
        selected_index = self._sample_index(self.video_dict[video_name],

                                            self.num_local_frames,
                                            self.num_ref_frames)

        # Read video frames
        frames = []
        depths = []
        masks = []
        for idx in selected_index:
            frame_list = self.frame_dict[video_name]
            img_path = os.path.join(self.video_root, video_name, frame_list[idx])
            img_bytes = self.file_client.get(img_path, 'img')
            img = imfrombytes(img_bytes, float32=False)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img)

            # Depth
            depth_path = os.path.join(self.depth_path, video_name, frame_list[idx].split('.')[0] + '.png')
            depth = Image.open(depth_path).convert('RGB').resize(self.size)

            frames.append(img)
            depths.append(depth)
            masks.append(all_masks[idx])

            if len(frames) == self.num_local_frames: # random reverse
                if random.random() < 0.5:
                    frames.reverse()
                    depths.reverse()
                    masks.reverse()
                
        frames_pil = GroupRandomHorizontalFlip()(frames)
        depths = GroupRandomHorizontalFlip()(depths)
        masks = GroupRandomHorizontalFlip()(masks)
        
        # Normalizate and convert to tensors
        frame_tensors = self._to_tensors(frames_pil) * 2.0 - 1.0
        depth_tensors = self._to_tensors(depths)
        mask_tensors = self._to_tensors(masks)

        batch = {
            "frames": frame_tensors,
            "depths": depth_tensors,
            "masks": mask_tensors,
        }
        return batch