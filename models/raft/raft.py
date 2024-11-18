import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock, AlternateCorrBlock
from .utils.utils import coords_grid, upflow8

class RAFT(nn.Module):
    def __init__(self):
        super(RAFT, self).__init__()
        args = argparse.ArgumentParser()
        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4
        args.small = False
        args.mixed_precision = False
        args.alternate_corr = False

        if 'dropout' not in args._get_kwargs():
            args.dropout = 0

        if 'alternate_corr' not in args._get_kwargs():
            args.alternate_corr = False
        
        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def initialize(self, raft_path):
        old_state_dict = torch.load(raft_path, map_location='cpu')
        new_state_dict = {}
        for name, param in old_state_dict.items():
            if 'module.' in name:
                name = name[7:]
            new_state_dict[name] = param
        self.load_state_dict(new_state_dict)
        self.requires_grad_(False)
        self.eval()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)


    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=True):
        """ Estimate optical flow between pair of frames """

        # image1 = 2 * (image1 / 255.0) - 1.0
        # image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        fmap1, fmap2 = self.fnet([image1, image2])
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        cnet = self.cnet(image1)
        net, inp = torch.split(cnet, [hdim, cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions

"""Modified from the one in ProPainter."""
class RAFTRunner(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.raft_model = RAFT()
        self.raft_model.initialize(model_path)
        self.raft_model = self.raft_model.to(device)

        for p in self.raft_model.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, gt_local_frames, iters=20, reverse_flow=False):
        if isinstance(gt_local_frames, list):
            tensor_frames = []
            for frame in gt_local_frames:
                tensor_frame = torch.from_numpy(np.array(frame).astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0)
                tensor_frames.append(tensor_frame)
            gt_local_frames = torch.cat(tensor_frames, dim=0).unsqueeze(0).cuda()
        
        b, l_t, c, h, w = gt_local_frames.size()

        with torch.no_grad():
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)
            
            if reverse_flow:
                _, gt_flows_forward = self.raft_model(gtlf_2, gtlf_1, iters=iters, test_mode=True)
            else:
                _, gt_flows_forward = self.raft_model(gtlf_1, gtlf_2, iters=iters, test_mode=True)
    
        gt_flows_forward = gt_flows_forward.view(b, l_t-1, 2, h, w)

        return gt_flows_forward
    
"""Extract 5-D dimension optical flow."""
class RAFTRunner3D(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.raft_model = RAFT()
        self.raft_model.initialize(model_path)
        self.raft_model = self.raft_model.to(device)

        for p in self.raft_model.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, gt_local_frames, iters=20):
        b, l_t, c, h, w = gt_local_frames.size()

        with torch.no_grad():
            gtlf_1 = gt_local_frames[:, :-1, :, :, :].reshape(-1, c, h, w)
            gtlf_2 = gt_local_frames[:, 1:, :, :, :].reshape(-1, c, h, w)

            _, gt_flows_forward = self.raft_model(gtlf_1, gtlf_2, iters=iters, test_mode=True)
            _, gt_flows_backward = self.raft_model(gtlf_2, gtlf_1, iters=iters, test_mode=True)

        
        gt_flows_forward = gt_flows_forward.view(b, l_t-1, 2, h, w)
        gt_flows_backward = gt_flows_backward.view(b, l_t-1, 2, h, w)

        return gt_flows_forward, gt_flows_backward
