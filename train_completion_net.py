"""
A script to train flow completion net.
"""
import os
import random
import numpy as np
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from runners.completion_net_train_runner import flow_completion_net_train_runner, depth_completion_net_train_runner

DEFAULT_NEGATIVE_PROMPTS = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"


# Functions for DDP
def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Setting up the process on rank {rank}.")

def cleanup():
    dist.destroy_process_group()

def setup_seed(seed):
    print(f'Seed: {seed}.')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    # Data configurations
    parser.add_argument("--video-path", type=str, default='', help="Path of videos")
    parser.add_argument("--shape-path", type=str, default='', help="Path of first frames' shapes")
    parser.add_argument("--depth-path", type=str, default='', help="Path of pre-extracted depth maps")
    parser.add_argument("--flow-path", type=str, default='', help="Path of pre-extracted optical flows")
    parser.add_argument("--annotation-path", type=str, default='', help="Path of the annotation file")
    parser.add_argument("--n-sample-frames", type=int, default=16, help="Number of sampled video frames")
    parser.add_argument("--output-fps", type=int, default=3)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--num-local-frames", type=int, default=10)
    parser.add_argument("--num-ref-frames", type=int, default=1)
    parser.add_argument("--load-flow", type=str, default=None)
    
    # Optimizer configurations
    parser.add_argument("--learning-rate",type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--adam-beta1", type=float, default=0, help="The beta1 parameter of optimizer")
    parser.add_argument("--adam-beta2", type=float, default=0.99, help="The beta2 parameter of optimizer")
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2, help="Weight decay of optimizer")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value of optimizer")

    
    # Training configurations
    parser.add_argument("--raft-checkpoint-path", type=str, default='', help='Local path of RAFT')
    parser.add_argument("--depth-estimator-checkpoint-path", type=str, default='', help="Local path of MiDas")
    parser.add_argument("--seed", type=int, default=23, help="Random seed")
    parser.add_argument("--master-port", type=str, default=str(2333), help="Master port for DDP")
    parser.add_argument("--dtype", type=str, default='fp32', help="Data dtype")
    parser.add_argument("--max-train-steps", type=int, default=9999999, help="Training steps")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--outdir", type=str, default='outputs/training/flow_completion_net', help='Output directory')
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of dataloader workers")
    parser.add_argument("--lr-scheduler", type=str, default='MultiStepLR', help="Type of learning-rate scheduler")
    parser.add_argument("--milestones", type=list, default=[300e3, 400e3, 500e3, 600e3], help="Milestone step to adjust the learning rate")
    parser.add_argument("--gamma", type=float, default=0.2, help="Gamma value to adjust the learning rate")
    parser.add_argument("--completion-net-in-channels", type=int, default=4, help="Input channel of the complention net")
    parser.add_argument("--completion-net-out-channels", type=int, default=3, help="Output channel of the complention net")
    parser.add_argument("--resume-completion-net", type=str, default=None, help="Perform resume training or not")
    parser.add_argument("--resume-step", type=int, default=0, help="Perform resume training or not")

    # Checkpointing configurations
    parser.add_argument("--checkpoint-freq", type=int, default=2000, help="Checkpoint frequency")
    parser.add_argument("--validation-freq", type=int, default=200, help="Validation frequency")
  
    # Loss configurations
    # NOTE: Flow completion
    parser.add_argument("--flow-loss-weight", type=float, default=0.25, help="Weight of flow loss")
    parser.add_argument("--warp-loss-weight", type=float, default=0.01, help="Weight of warp loss")
    # NOTE: Depth completion
    parser.add_argument("--depth-loss-weight", type=float, default=1.0, help="Weight of depth loss")
    parser.add_argument("--edge-loss-weight", type=float, default=1.0, help="Weight of edge loss")
    parser.add_argument("--combined-edge-loss-weight", type=float, default=5.0, help="Weight of combined edge loss")
    
    args = parser.parse_args()
    
    return args


# Main function for DDP
def main_worker(rank, world_size, args):
    setup(rank=rank, world_size=world_size, master_port=args.master_port)
    depth_completion_net_train_runner(rank=rank,
                                          args=args)


if __name__ == "__main__":
    # Set up basic configurations
    args = parse_args()
    setup_seed(args.seed)
    args.world_size = torch.cuda.device_count()
    if args.world_size > 1:
        args.DDP = True
    else:
        args.DDP = False
    
    # Execute the train runner
    # Use DDP for multiple-GPU training
    if args.DDP:
        mp.spawn(main_worker, nprocs=args.world_size, args=(args.world_size, args))
    # Use single GPU for training
    else:
        depth_completion_net_train_runner(rank=0,
                                              args=args)
    
