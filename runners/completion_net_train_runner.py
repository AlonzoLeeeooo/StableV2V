import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from datasets.youtube_vos import YouTubeVOSDepthDataset
from models.depth_completion_net.rfc_net import RecurrentDepthCompleteNet
from models.canny.canny_filter import Canny
from utils.utils import count_params
from utils.loss_utils import DepthLoss, EdgeLossForDepthCompletion
from utils.lr_scheduler_utils import MultiStepRestartLR

canny_func = Canny(sigma=(2,2), low_threshold=0.1, high_threshold=0.2)
depth_loss_func = DepthLoss()
edge_loss_func = EdgeLossForDepthCompletion()


def get_edges(x): 
    # (b, t, 2, H, W)
    b, t, _, h, w = x.shape
    x = x.view(-1, 2, h, w)
    x_gray = (x[:, 0, None] ** 2 + x[:, 1, None] ** 2) ** 0.5
    if x_gray.max() < 1:
        x_gray = x_gray * 0
    else:
        x_gray = x_gray / x_gray.max()
        
    magnitude, edges = canny_func(x_gray.float())
    edges = edges.view(b, t, 1, h, w)
    return edges

def get_edges_from_depth(x): 
    b, t, _, h, w = x.shape
    x = x.view(-1, 3, h, w)
    x_gray = (x[:, 0, None] ** 2 + x[:, 1, None] ** 2) ** 0.5
    if x_gray.max() < 1:
        x_gray = x_gray * 0
    else:
        x_gray = x_gray / x_gray.max()
        
    magnitude, edges = canny_func(x_gray.float())
    edges = edges.view(b, t, 1, h, w)
    return edges


def bar(prg):
    br = '|' + '█' * prg + ' ' * (25 - prg) + '|'
    return br            
            
            
def depth_completion_net_train_runner(rank,
                                      args):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if args.dtype == "fp32":
        data_dtype = torch.float32
    elif args.dtype == "fp16":
        data_dtype = torch.float16
    elif args.dtype == "bf16":
        data_dtype = torch.bfloat16
    os.makedirs(os.path.join(args.outdir, 'tensorboard'), exist_ok=True)
    if rank == 0:
        tensorboard_writter = SummaryWriter(os.path.join(args.outdir, 'tensorboard'))
    
    # Define the models
    completion_net = RecurrentDepthCompleteNet(in_channels=args.completion_net_in_channels,
                                              out_channels=args.completion_net_out_channels)
    
    # Perform resume training or not
    if args.resume_completion_net is not None:
        completion_net.load_state_dict(torch.load(args.resume_completion_net, map_location='cpu'))
        print(f"Resume model checkpoint from {args.resume_completion_net}, step {args.resume_step}.")
    completion_net = completion_net.to(device)
    
    if args.DDP:
        completion_net = DistributedDataParallel(completion_net, device_ids=[rank], find_unused_parameters=True)
    else:
        pass
    
    # NOTE: We do not use online depth estimation anymore, since it is too slow
    # Define the depth estimator (MiDas)
    # depth_estimator = DepthMidas(model_path=args.depth_estimator_checkpoint_path,
    #                              device=device)
    # depth_estimator.model.eval()
    # depth_estimator.model.requires_grad_(False)
    
    # Define the datasets
    train_dataset = YouTubeVOSDepthDataset(args)
    
    # Define the dataloaders
    if args.DDP:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank)
    else:
        train_sampler = DistributedSampler(train_dataset, num_replicas=1, rank=0)
    
    train_loader = DataLoader(train_dataset,
                              shuffle=False,
                              pin_memory=False,
                              batch_size=args.batch_size // args.world_size,
                              num_workers=args.num_workers,
                              sampler=train_sampler)
    
    
    # Define the optimizer
    parameters_list = []
    for name, para in completion_net.named_parameters():
        para.requires_grad_(True)
        parameters_list.append(para)
    count_params(parameters_list)
    optimizer = torch.optim.AdamW(completion_net.parameters(),
                                  lr=args.learning_rate,
                                  betas=(args.adam_beta1, args.adam_beta2))
    
    
    # Define the learning-rate scheduler
    if args.lr_scheduler in ['MultiStepLR', 'MultiStepRestartLR']:
        lr_scheduler = MultiStepRestartLR(
            optimizer=optimizer,
            milestones=args.milestones,
            gamma=args.gamma
        )
    
    # Define progressive bar
    if rank == 0:
        print("Start training...")
        progress_bar = tqdm(range(args.max_train_steps))
        
    # Training loop
    step = args.resume_step
    completion_net.train()
    for epoch in range(args.epochs):
        for index, batch in enumerate(train_loader):
            step += 1
            
            # For depth estimation, we extract 10 frames to obtain 10-frame depth maps
            frames, masks, depths, shapes = batch['frames'], batch['masks'], batch['depths'], batch['shapes']
            
            b, t, c, h, w = frames.size()
            gt_local_frames = frames[:, :args.num_local_frames, ...].to(device)
            local_masks = masks[:, :args.num_local_frames, ...].to(device)
            gt_depth_maps = depths[:, :args.num_local_frames, ...].to(device)
            # We use repeated the first-frame shape as input condition
            shapes = torch.cat([shapes[:, :1, :, :, :]] * args.num_local_frames, dim=1).to(device)
        
            
            # Ground truth Canny edge
            gt_edges = get_edges_from_depth(gt_depth_maps)
            
            # Forward
            optimizer.zero_grad()
            if isinstance(completion_net, DistributedDataParallel):
                pred_depth_maps, pred_edges = completion_net.module(gt_depth_maps.to(device), local_masks.to(device), shapes.to(device))
            else:
                pred_depth_maps, pred_edges = completion_net.forward(gt_depth_maps.to(device), local_masks.to(device), shapes.to(device))
            
            # Compute losses
            depth_loss = depth_loss_func(pred_depth_maps, gt_depth_maps.to(device), local_masks, gt_local_frames)
            depth_loss = depth_loss * args.depth_loss_weight
            edge_loss = edge_loss_func(pred_edges, gt_edges.to(device), local_masks, args.combined_edge_loss_weight)
            edge_loss = edge_loss * args.edge_loss_weight
            tensorboard_writter.add_scalar('train/depth_loss', depth_loss.item(), global_step=step)
            tensorboard_writter.add_scalar('train/edge_loss', edge_loss.item(), global_step=step)
            
            loss = depth_loss + edge_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Update progressive bar
            if rank == 0:
                progress_bar.update(1)
                progress_bar.set_description((f"epoch: {epoch + 1}, "
                                            f"depth: {depth_loss.item():.3f}, "
                                            f"edge: {edge_loss.item():.3f}, "
                                            f"lr: {lr_scheduler.get_lr()}"))
            
            # Save checkpoint & visualization
            if step % args.checkpoint_freq == 0:
                # Save the trained model checkpoint
                os.makedirs(os.path.join(args.outdir, 'checkpoints'), exist_ok=True)
                save_path = os.path.join(args.outdir, 'checkpoints', f"{step}.ckpt")
                torch.save(completion_net.state_dict(), save_path)
                
            if step % args.validation_freq == 0:
                # Visualization            
                # Depths
                gt_depth_maps_cpu = gt_depth_maps[0].cpu()
                t = args.num_local_frames - 2
                masked_depth_maps_cpu = (gt_depth_maps_cpu[t] * (1 - local_masks[0][t].cpu())).to(gt_depth_maps_cpu)
                pred_depth_maps_cpu = pred_depth_maps[0].cpu()
                shape_cpu = shapes[0][t].cpu()
                shape_cpu_3_ch = torch.cat([shape_cpu] * 3, dim=0)

                depth_results = torch.cat([gt_depth_maps_cpu[t], masked_depth_maps_cpu, shape_cpu_3_ch, pred_depth_maps_cpu[t]], dim=2)
                tensorboard_writter.add_image('depth_maps', depth_results, step)
                
                # Edges
                gt_edges_cpu = gt_edges[0].cpu()
                masked_edges_cpu = (gt_edges_cpu[t] * (1 - local_masks[0][t].cpu())).to(gt_edges_cpu)
                pred_edges_cpu = pred_edges[0].cpu()

                edge_results = torch.cat([gt_edges_cpu[t], masked_edges_cpu, shape_cpu, pred_edges_cpu[t]], dim=2)
                tensorboard_writter.add_image('edges', edge_results, step)
    
