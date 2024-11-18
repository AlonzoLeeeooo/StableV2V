<div align="center">

# StableV2V: Stablizing Shape Consistency in Video-to-Video Editing

Chang Liu, Rui Li, Kaidong Zhang, Yunwei Lan, Dong Liu

[[`Paper`]]() / [[`Project`]](https://alonzoleeeooo.github.io/StableV2V/) / [[`Huggingface`]](https://huggingface.co/AlonzoLeeeooo/StableV2V) / [[`Dataset`]](https://huggingface.co/datasets/AlonzoLeeeooo/DAVIS-Edit)
</div>

<!-- omit in toc -->
# Table of Contents
- [<u>1. News</u>](#news)
- [<u>2. To-Do Lists</u>](#to-do-lists)
- [<u>3. Overview of StableV2V</u>](#overview-of-stablev2v)
- [<u>4. Code Structure</u>](#code-structure)
- [<u>5. Prerequisites</u>](#prerequisites)
- [<u>6. Inference of StableV2V</u>](#inference-of-stablev2v)
- [<u>7. Training the Shape-guided Depth Refinement Network</u>](#training-of-the-shape-guided-depth-refinement-network)
- [<u>8. Citation</u>](#citation)
- [<u>9. Star History</u>](#star-history)

If you have any questions about this work, please feel free to [start a new issue](https://github.com/AlonzoLeeeooo/StableV2V/issues/new) or [propose a PR](https://github.com/AlonzoLeeeooo/StableV2V/pulls).


<!-- omit in toc -->
# News
- [Nov. 17th] We updated our [project page](https://alonzoleeeooo.github.io/StableV2V/).

<!-- omit in toc -->
# To-Do List
- [x] Update the codebase of `StableV2V`
- [ ] Upload the required model weights of `StableV2V`
- [ ] Upload the curated testing benchmark `DAVIS-Edit`
- Regular Maintainence

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)



<!-- omit in toc -->
# Overview of StableV2V
StableV2V presents a novel paradigm to perform video editing in a shape-consistent manner, especially handling the editing scenarios when user prompts cause significant shape changes to the edited contents.
Besides, StableV2V shows superior flexibility in handling a wide series of down-stream applications, considering various user prompts from different modalities.

<div align="center">
  <video width="500" src="assets/github-teasor-comparison.mp4" autoplay loop muted></video>
  <video width="500" src="assets/github-teasor-applications.mp4" autoplay loop muted></video>
</div>



[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Code Structure
```
StableV2V
├── LICENSE
├── README.md
├── assets
├── datasets                       <----- Code of datasets for training of the depth refinement network
├── models                         <----- Code of model definitions in different components
├── runners                        <----- Code of engines to run different components
├── inference.py                   <----- Script to inference StableV2V
├── train_completion_net.py        <----- Script to train the shape-guided depth completion network
└── utils                          <----- Code of toolkit functions
```

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Prerequisites
<!-- omit in toc -->
## 1. Install the Dependencies
We offer an one-click command line to install all the dependencies that the code requires.
Specifically, you can execute:
```bash
pip install -r requirements.txt
```

<!-- omit in toc -->
## 2. Pre-trained Model Weights
Before you start the inference process, you need to prepare the model weights that `StableV2V` requires.

> [!NOTE]
> Currently, we are uploading the pre-trained model weights to our [HuggingFace repo](https://huggingface.co/AlonzoLeeeooo/StableV2V), so that users can get access to all weights in the same repo.
> Before that, you may refer to the official links in the following table:

|Model|Component|Link|
|-|-|-|
|Paint-by-Example|PFE|[`Fantasy-Studio/Paint-by-Example`](https://huggingface.co/Fantasy-Studio/Paint-by-Example)|
|InstructPix2Pix|PFE|[`timbrooks/instruct-pix2pix`](https://huggingface.co/timbrooks/instruct-pix2pix)|
|SD Inpaint|PFE|[`botp/stable-diffusion-v1-5-inpainting`](https://huggingface.co/botp/stable-diffusion-v1-5-inpainting)|
|ControlNet + SD Inpaint|PFE|ControlNet models at [`lllyasviel`](https://huggingface.co/lllyasviel)|
|AnyDoor|PFE|[`xichenhku/AnyDoor`](https://huggingface.co/spaces/xichenhku/AnyDoor/tree/main)|
|RAFT|ISA|[`Google Drive`](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)|
|MiDaS|ISA|[`Link`](https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt)|
|U2-Net|ISA|[`Link`](https://huggingface.co/AlonzoLeeeooo/LaCon/resolve/main/data-preprocessing/u2net.pth)|
|Depth Refinement Network|ISA|[`Link`](https://huggingface.co/AlonzoLeeeooo/StableV2V)|
|SD v1.5|CIG|[`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/)|
|ControlNet (depth)|CIG|[`lllyasviel/control_v11f1p_sd15_depth`](https://huggingface.co/lllyasviel/control_v11f1p_sd15_depth)|
|Ctrl-Adapter|CIG|[`hanlincs/Ctrl-Adapter`](https://huggingface.co/hanlincs/Ctrl-Adapter) (`i2vgenxl_depth`)|
|I2VGen-XL|CIG|[`ali-vilab/i2vgen-xl`](https://huggingface.co/ali-vilab/i2vgen-xl)|

Once you downloaded all the model weights, put them in the `checkpoints` folder.

> [!NOTE]
> If your network environment can get access to HuggingFace, you can directly use the HuggingFace repo ID to download the models.
> Otherwise we highly recommend you to prepare the model weights locally.

Specfically, make sure you modify the configuration file of `AnyDoor` at `models/anydoor/configs/anydoor.yaml` with the path of DINO-v2 pre-trained weights:
```
(at line 83)
cond_stage_config:
  target: models.anydoor.ldm.modules.encoders.modules.FrozenDinoV2Encoder
  weight: /path/to/dinov2_vitg14_pretrain.pth
```



[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)



<!-- omit in toc -->
# Inference of StableV2V

You may refer to the following command line to run `StableV2V`:
```bash
python inference.py --raft-checkpoint-path checkpoints/raft-things.pth --midas-checkpoint-path checkpoints/dpt_swin2_large_384.pt --u2net-checkpoint-path checkpoints/u2net.pth  --stable-diffusion-checkpoint-path stable-diffusion-v1-5/stable-diffusion-v1-5 --controlnet-checkpoint-path lllyasviel/control_v11f1p_sd15_depth --i2vgenxl-checkpoint-path ali-vilab/i2vgen-xl --ctrl-adapter-checkpoint-path hanlincs/Ctrl-Adapter --completion-net-checkpoint-path checkpoints/depth-refinement/50000.ckpt --image-editor-type paint-by-example --image-editor-checkpoint-path /path/to/image/editor --source-video-frames examples/frames/bear --external-guidance examples/reference-images/raccoon.jpg --prompt "a raccoon" --outdir results
```

For detailed illustrations of the arguments, please refer to the table below:
|Argument|Default Setting|Required or Not|Explanation|
|-|-|-|-|
|Model arguments|-|-|-|
|`--image-editor-type`|-|Yes|Argument to define the image editor type.|
|`--image-editor-checkpoint-path`|-|Yes|Path of model weights for the image editor, required by PFE.|
|`--raft-checkpoint-path`|`checkpoints/raft-things.pth`|Yes|Path of model weights for RAFT, required by ISA.|
|`--midas-checkpoint-path`|`checkpoints/dpt_swin2_large_382.pt`|Yes|Path of model weights for MiDaS, required by ISA.|
|`--u2net-checkpoint-path`|`checkpoints/u2net.pth`|Yes|Path of model weights for U2-Net, required by ISA to obtain the segmentation masks of video frames (will be replaced by SAM in near future)|
|`--stable-diffusion-checkpoint-path`|`stable-diffusion-v1-5/stable-diffusion-v1-5`|Yes|Path of model weights for SD v1.5, required by CIG.|
|`--controlnet-checkpoint-path`|`lllyasviel/control_v11f1p_sd15_depth`|Yes|Path of model weights for ControlNet (depth) required by CIG.|
|`--ctrl-adapter-checkpoint-path`|`hanlincs/Ctrl-Adapter`|Yes|Path of model weights for Ctrl-Adapter, required by CIG.|
|`--i2vgenxl-checkpoint-path`|`ali-vilab/i2vgen-xl`|Yes|Path of model weights for I2VGen-XL, required by CIG.|
|`--completion-checkpoint-path`|`checkpoints/depth-refinement/50000.ckpt`|Yes|Path of model weights for I2VGen-XL, required by CIG.|
|Input Arguments|-|-|-|
|`--source-video-frames`|-|Yes|Path of input video frames.|
|`--prompt`|-|Yes|Text prompt of the edited video.|
|`--external-guidance`|-|Yes|External inputs for the image editors if you use `Paint-by-Example`, `InstructPix2Pix`, and `AnyDoor`.|
|`--outdir`|`results`|Yes|Path of output directory.|
|`--edited-first-frame`|-|No|Path of customized first edited frame, where the image editor will not be used if this argument is configured.|
|`--input-condition`|-|No|Path of cusromzied depth maps. We directly extract depth maps from the source video frames with `MiDaS` if this argument is not configured|
|`--input-condition`|-|No|Path of cusromzied depth maps. We directly extract depth maps from the source video frames with `MiDaS` if this argument is not configured.|
|`--reference-masks`|-|No|Path of segmentation masks of the reference image, required by `AnyDoor`. We will automatically extract segmentation mask from the reference image if this argument is not configured.|
|`--image-guidance-scale`|1.0|No|Hyper-parameter required by InstructPix2Pix.|
|`--kernel-size`|9|No|Kernel size of the binary dilation operation, to make sure that the pasting processes cover the regions of edited contents.|
|`--dilation-iteration`|1|No|Iteration for binary dilation operation.|
|`--guidance-scale`|9.0|No|Classifier-free guidance scale.|
|`--mixed-precision`|bf16|No|Precision of models in StableV2V.|
|`--n-sample-frames`|16|No|Number of video frames of the edited video.|
|`--seed`|42|No|Random seed.|

> [!NOTE]
> Some specific points that you may pay additional attentions to while inferencing:
>
> 1. By configuring `--image-editor-checkpoint-path`, the path will be automatically delievered to the corresponding editor according to your `--image-editor-type`. So please do not be worried about some extra arguments in the codebase.
> 2. If you are using `Paint-by-Example`, `InstructPix2Pix`, `AnyDoor`, you are required to configure the `--external-guidance` argument, which corresponds to reference image and user instruction accordingly.
> 3. Our method does not currently support `xformers`, which might cause artifacts in the produced results. Such issue might be fixed in the future if possible.


[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)





<!-- omit in toc -->
# Training of the Shape-guided Depth Refinement Network
We will update the instructions of shape-guided depth refinement network soon.


[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)



<!-- omit in toc -->
# Citation
We will update the BibTeX reference as soon as our arXiv paper is announced.



[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Star History

<p align="center">
    <a href="hhttps://api.star-history.com/svg?repos=alonzoleeeooo/StableV2V&type=Date" target="_blank">
        <img width="550" src="https://api.star-history.com/svg?repos=alonzoleeeooo/StableV2V&type=Date" alt="Star History Chart">
    </a>
</p>

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)
