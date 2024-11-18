import random
import argparse 
from PIL import Image, ImageOps
from typing import List, Union
from io import BytesIO
import numpy as np

import torch
from torchvision import transforms

from transformers import PretrainedConfig
from diffusers.image_processor import VaeImageProcessor


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
        

def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("total_trainable_params_count is: ", total_trainable_params_count)


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def enable_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == False:
            #print(name, p.requires_grad)
            p.requires_grad = True


def print_trainable_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            print(name, p.requires_grad)


def print_disabled_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == False:
            print(name, p.requires_grad)


def images_to_gif_bytes(images: List, duration: int = 1000) -> bytes:
    with BytesIO() as output_buffer:
        # Save the first image
        images[0].save(output_buffer,
                       format='GIF',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration,
                       loop=0)  # 0 means the GIF will loop indefinitely

        # Get the byte array from the buffer
        gif_bytes = output_buffer.getvalue()

    return gif_bytes


def save_as_gif(images: List, file_path: str, duration: int = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_gif_bytes(images, duration))


def save_concatenated_gif(single_image, output_gif_path, image_list_arrays, duration=500):

    # Ensure all lists are the same length
    if not all(len(lst) == len(image_list_arrays[0]) for lst in image_list_arrays):
        raise ValueError("All image lists must have the same number of elements")

    # Create a list to hold the concatenated frames
    concatenated_frames = []

    # Loop through each index in the list (assuming all lists have the same length)
    for index in range(len(image_list_arrays[0])):
        # Start with the single image
        images = [single_image] + [lst[index] for lst in image_list_arrays]

        # Calculate total width and max height
        total_width = sum(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)

        # Create a new image with the calculated dimensions
        new_image = Image.new('RGB', (total_width, max_height))

        # Paste each image into the new image
        current_x = 0
        for img in images:
            new_image.paste(img, (current_x, 0))
            current_x += img.size[0]
        
        concatenated_frames.append(new_image)

    # Save the final sequence of frames as a new GIF
    concatenated_frames[0].save(output_gif_path, save_all=True, append_images=concatenated_frames[1:], optimize=False, duration=duration, loop=0)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
        
        
def center_crop_and_resize(img, output_size=(512, 512)):
    # Load the image
    #img = Image.open(image_path)
    
    # Calculate the aspect ratio of the output image
    aspect_ratio = output_size[0] / output_size[1]
    
    # Get the current size of the image
    original_width, original_height = img.size
    
    # Calculate the aspect ratio of the original image
    original_aspect_ratio = original_width / original_height
    
    # Determine the dimensions to which the image needs to be resized before cropping
    if original_aspect_ratio > aspect_ratio:
        # Image is wider than the desired aspect ratio; resize based on height
        new_height = output_size[1]
        new_width = int(new_height * original_aspect_ratio)
    else:
        # Image is taller than the desired aspect ratio; resize based on width
        new_width = output_size[0]
        new_height = int(new_width / original_aspect_ratio)
    
    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate the cropping box
    left = (new_width - output_size[0]) / 2
    top = (new_height - output_size[1]) / 2
    right = (new_width + output_size[0]) / 2
    bottom = (new_height + output_size[1]) / 2
    
    # Crop the center
    img_cropped = img_resized.crop((left, top, right, bottom))
    
    return img_cropped


image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def image_to_tensor(img):
    
    with torch.no_grad():
        if img.mode != "RGB":
            img = img.convert("RGB")

        image = image_transforms(img)#.to(accelerator.device)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.shape[0] > 3:
            image = image[:3, :, :]

    return image


# NOTE: Newly defined functions
def binarize_tensor(tensor):
    with torch.no_grad():
        tensor[tensor > 0.5] = 1
        tensor[tensor < 0.5] = 0
        
        return tensor
    
    
import argparse 
from PIL import Image
from typing import List, Union
from io import BytesIO

import torch
from torchvision import transforms

from transformers import PretrainedConfig



def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
        

def count_params(params):
    total_trainable_params_count = 0 
    for p in params:
        total_trainable_params_count += p.numel()
    print("Total trainable parameters: ", total_trainable_params_count)


def batch_to_device(batch, device):
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    return batch


def disable_grads(model):
    for p in model.parameters():
        p.requires_grad = False


def enable_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == False:
            #print(name, p.requires_grad)
            p.requires_grad = True


def print_trainable_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == True:
            print(name, p.requires_grad)


def print_disabled_grads(model):
    for name, p in model.named_parameters():
        if p.requires_grad == False:
            print(name, p.requires_grad)


def images_to_gif_bytes(images: List, duration: int = 1000) -> bytes:
    with BytesIO() as output_buffer:
        # Save the first image
        images[0].save(output_buffer,
                       format='GIF',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration,
                       loop=0)  # 0 means the GIF will loop indefinitely

        # Get the byte array from the buffer
        gif_bytes = output_buffer.getvalue()

    return gif_bytes


def save_as_gif(images: List, file_path: str, duration: int = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_gif_bytes(images, duration))


def save_concatenated_gif(single_image, output_gif_path, image_list_arrays, duration=500):

    # Ensure all lists are the same length
    if not all(len(lst) == len(image_list_arrays[0]) for lst in image_list_arrays):
        raise ValueError("All image lists must have the same number of elements")

    # Create a list to hold the concatenated frames
    concatenated_frames = []

    # Loop through each index in the list (assuming all lists have the same length)
    for index in range(len(image_list_arrays[0])):
        # Start with the single image
        images = [single_image] + [lst[index] for lst in image_list_arrays]

        # Calculate total width and max height
        total_width = sum(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)

        # Create a new image with the calculated dimensions
        new_image = Image.new('RGB', (total_width, max_height))

        # Paste each image into the new image
        current_x = 0
        for img in images:
            new_image.paste(img, (current_x, 0))
            current_x += img.size[0]
        
        concatenated_frames.append(new_image)

    # Save the final sequence of frames as a new GIF
    concatenated_frames[0].save(output_gif_path, save_all=True, append_images=concatenated_frames[1:], optimize=False, duration=duration, loop=0)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")
        
        
def center_crop_and_resize(img, output_size=(512, 512)):
    # Load the image
    #img = Image.open(image_path)
    
    # Calculate the aspect ratio of the output image
    aspect_ratio = output_size[0] / output_size[1]
    
    # Get the current size of the image
    original_width, original_height = img.size
    
    # Calculate the aspect ratio of the original image
    original_aspect_ratio = original_width / original_height
    
    # Determine the dimensions to which the image needs to be resized before cropping
    if original_aspect_ratio > aspect_ratio:
        # Image is wider than the desired aspect ratio; resize based on height
        new_height = output_size[1]
        new_width = int(new_height * original_aspect_ratio)
    else:
        # Image is taller than the desired aspect ratio; resize based on width
        new_width = output_size[0]
        new_height = int(new_width / original_aspect_ratio)
    
    # Resize the image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Calculate the cropping box
    left = (new_width - output_size[0]) / 2
    top = (new_height - output_size[1]) / 2
    right = (new_width + output_size[0]) / 2
    bottom = (new_height + output_size[1]) / 2
    
    # Crop the center
    img_cropped = img_resized.crop((left, top, right, bottom))
    
    return img_cropped


image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def image_to_tensor(img):
    
    with torch.no_grad():
        if img.mode != "RGB":
            img = img.convert("RGB")

        image = image_transforms(img)#.to(accelerator.device)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        if image.shape[0] > 3:
            image = image[:3, :, :]

    return image


# NOTE: Newly defined functions
def binarize_tensor(tensor):
    with torch.no_grad():
        tensor[tensor > 0.5] = 1
        tensor[tensor < 0.5] = 0
        
        return tensor
    
# The following utilities are taken and adapted from
# https://github.com/ali-vilab/i2vgen-xl/blob/main/utils/transforms.py.
def _convert_pt_to_pil(image: Union[torch.Tensor, List[torch.Tensor]]):
    if isinstance(image, list) and isinstance(image[0], torch.Tensor):
        image = torch.cat(image, 0)

    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image_numpy = VaeImageProcessor.pt_to_numpy(image)
        image_pil = VaeImageProcessor.numpy_to_pil(image_numpy)
        image = image_pil

    return image
    
    
class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [L, C, H, W]
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img
    
class GroupRandomHorizontalFlowFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group, flowF_group, flowB_group):
        v = random.random()
        if v < 0.5:
            ret_img = [
                img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group
            ]
            ret_flowF = [ff[:, ::-1] * [-1.0, 1.0] for ff in flowF_group]
            ret_flowB = [fb[:, ::-1] * [-1.0, 1.0] for fb in flowB_group]
            return ret_img, ret_flowF, ret_flowB
        else:
            return img_group, flowF_group, flowB_group


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if is_flow:
                for i in range(0, len(ret), 2):
                    # invert flow pixel values when flipping
                    ret[i] = ImageOps.invert(ret[i])
            return ret
        else:
            return img_group
    
    
def _resize_bilinear(
    image: Union[torch.Tensor, List[torch.Tensor], Image.Image, List[Image.Image]], resolution: tuple[int, int]
):
    # First convert the images to PIL in case they are float tensors (only relevant for tests now).
    image = _convert_pt_to_pil(image)

    if isinstance(image, list):
        image = [u.resize(resolution, Image.BILINEAR) for u in image]
    else:
        image = image.resize(resolution, Image.BILINEAR)
    return image

def _get_add_time_ids(original_size, crops_coords_top_left, target_size, dtype):
     
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids