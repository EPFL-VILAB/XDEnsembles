
import os, sys, math, random, itertools, functools
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as util_checkpoint
from torchvision import models

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import get_task, task_map, get_model, Task, RealityTask

from modules.percep_nets import DenseNet, Dense1by1Net, DenseKernelsNet, DeepNet, BaseNet, WideNet, PyramidNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet, UNetOld, UNetOld2, UNetReshade, UNet_w
from modules.resnet import ResNetClass

from fire import Fire
import IPython


### 

pretrained_transfers = {

    # percep models used in consistency training
    ('normal', 'principal_curvature'):
        (lambda: Dense1by1Net(), f"{MODELS_DIR}/perceps/normal2curvature.pth"),
    ('normal', 'depth_zbuffer'):
        (lambda: UNetDepth(), f"{MODELS_DIR}/perceps/normal2zdepth_zbuffer.pth"),
    ('normal', 'sobel_edges'):
        (lambda: UNet(out_channels=1, downsample=4).cuda(), f"{MODELS_DIR}/perceps/normal2edges2d.pth"),
    ('normal', 'reshading'):
        (lambda: UNetReshade(downsample=5), f"{MODELS_DIR}/perceps/normal2reshade.pth"),
    ('normal', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/perceps/normal2keypoints3d.pth"),
    ('normal', 'keypoints2d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/perceps/normal2keypoints2d.pth"),
    ('normal', 'edge_occlusion'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/perceps/normal2edge_occlusion.pth"),

    ('depth_zbuffer', 'sobel_edges'):
        (lambda: UNet(downsample=4, in_channels=1, out_channels=1).cuda(), f"{MODELS_DIR}/perceps/depth_zbuffer2sobel_edges.pth"),
    ('depth_zbuffer', 'principal_curvature'):
        (lambda: UNet(downsample=4, in_channels=1), f"{MODELS_DIR}/perceps/depth_zbuffer2principal_curvature.pth"),
    ('depth_zbuffer', 'reshading'):
        (lambda: UNetReshade(downsample=5, in_channels=1), f"{MODELS_DIR}/perceps/depth_zbuffer2reshading.pth"),
    ('depth_zbuffer', 'keypoints3d'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{MODELS_DIR}/perceps/depth_zbuffer2keypoints3d.pth"),
    ('depth_zbuffer', 'keypoints2d'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{MODELS_DIR}/perceps/depth_zbuffer2keypoints2d.pth"),
    ('depth_zbuffer', 'edge_occlusion'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{MODELS_DIR}/perceps/depth_zbuffer2edge_occlusion.pth"),
    ('depth_zbuffer', 'normal'):
        (lambda: UNet(in_channels=1, downsample=6), f"{MODELS_DIR}/perceps/depth2normal.pth"),

    ('reshading', 'depth_zbuffer'):
        (lambda: UNetReshade(downsample=5, out_channels=1), f"{MODELS_DIR}/perceps/reshading2depth_zbuffer.pth"),
    ('reshading', 'keypoints2d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/perceps/reshading2keypoints2d.pth"),
    ('reshading', 'edge_occlusion'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/perceps/reshading2edge_occlusion.pth"),
    ('reshading', 'normal'):
        (lambda: UNet(downsample=4), f"{MODELS_DIR}/perceps/reshading2normal.pth"),
    ('reshading', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/perceps/reshading2keypoints3d.pth"),
    ('reshading', 'sobel_edges'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/perceps/reshading2sobel_edges.pth"),
    ('reshading', 'principal_curvature'):
        (lambda: UNet(downsample=5), f"{MODELS_DIR}/perceps/reshading2principal_curvature.pth"),

    ('normal', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),
    ('reshading', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),
    ('depth_zbuffer', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),

    # rgb->mid domain
    ('rgb', 'sobel_edges'):
        (lambda: sobel_kernel, None),
    ('rgb', 'binarized'):
        (lambda: binarized_kernel, None),
    ('rgb', 'laplace_edges'):
        (lambda: laplace_kernel, None),
    ('rgb', 'gauss'):
        (lambda: gauss_kernel, None),
    ('rgb', 'emboss'):
        (lambda: emboss_kernel, None),
    ('rgb', 'grey'):
        (lambda: greyscale, None),
    ('rgb', 'wav'):
        (lambda: wav_kernel, None),
    ('rgb', 'sharp'):
        (lambda: sharp_kernel, None),
    ('rgb', 'emboss4d'):
        (lambda: emboss4d_kernel, None),
    
    # rgb->target domain
    ('rgb', 'normal'):
        (lambda: UNet(out_channels=6).cuda(),  f"{MODELS_DIR}/wo_cons/rgb2normal.pth"),
    ('rgb', 'normal_t0'):
        (lambda: UNet(out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/rgb2normal.pth"),
    ('rgb', 'reshading'):
        (lambda: UNet(downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/rgb2reshade.pth"),
    ('rgb', 'reshading_t0'):
        (lambda: UNet(downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/rgb2reshade.pth"),
    ('rgb', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/rgb2depth.pth"),
    ('rgb', 'depth_zbuffer_t0'):
        (lambda: UNet(downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/rgb2depth.pth"),

    # mid domain->target domain models
    ('sobel_edges', 'reshading'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/edge2reshade.pth"),
    ('sobel_edges', 'reshading_t0'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/edge2reshade.pth"),

    ('laplace_edges', 'reshading'):
        (lambda: UNet(in_channels=1, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/laplace2reshade.pth"),
    ('laplace_edges', 'reshading_t0'):
        (lambda: UNet(in_channels=1, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/laplace2reshade.pth"),

    ('laplace_edges', 'normal'):
        (lambda: UNet(in_channels=1, downsample=6, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/laplace2normal.pth"),
    ('laplace_edges', 'depth_zbuffer'):
        (lambda: UNet(in_channels=1, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/laplace2depth.pth"),

    ('gauss', 'reshading'):
        (lambda: UNet(in_channels=3, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/gauss2reshade.pth"),
    ('gauss', 'reshading_t0'):
        (lambda: UNet(in_channels=3, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/gauss2reshade.pth"),
    ('gauss', 'normal'):
        (lambda: UNet(in_channels=3, downsample=6, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/gauss2normal.pth"),
    ('gauss', 'depth_zbuffer'):
        (lambda: UNet(in_channels=3, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/gauss2depth.pth"),

    ('emboss4d', 'reshading'):
        (lambda: UNet(in_channels=4, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/emboss2reshade.pth"),
    ('emboss4d', 'reshading_t0'):
        (lambda: UNet(in_channels=4, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/emboss2reshade.pth"),
    
    ('emboss4d', 'depth_zbuffer'):
        (lambda: UNet(in_channels=4, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/emboss2depth.pth"),
    ('emboss4d', 'depth_zbuffer_0'):
        (lambda: UNet(in_channels=4, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/emboss2depth.pth"),
    
    ('emboss4d', 'normal'):
        (lambda: UNet(in_channels=4, downsample=6, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/emboss2normal.pth"), 
    ('emboss4d', 'normal_t0'):
        (lambda: UNet(in_channels=4, downsample=6, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/emboss2normal.pth"), 

    ('sharp', 'reshading'):
        (lambda: UNet(in_channels=3, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/sharp2reshade.pth"),
    ('sharp', 'depth_zbuffer'):
        (lambda: UNet(in_channels=3, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/sharp2depth.pth"),
    ('sharp', 'normal'):
        (lambda: UNet(in_channels=3, downsample=6, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/sharp2normal.pth"),

    ('grey', 'reshading'):
        (lambda: UNet(in_channels=1, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/grey2reshade.pth"),
    ('grey', 'reshading_t0'):
        (lambda: UNet(in_channels=1, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/grey2reshade.pth"),

    ('wav', 'reshading'):
        (lambda: UNet(in_channels=12, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/wavelet2reshade.pth"),
    ('wav', 'reshading_t0'):
        (lambda: UNet(in_channels=12, downsample=5, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/wavelet2reshade.pth"),

    ('sobel_edges', 'normal'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/edge2norm.pth"),
    ('sobel_edges', 'normal_t0'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/edge2norm.pth"),

    ('grey', 'normal'):
        (lambda: UNet(downsample=6, in_channels=1, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/grey2normal.pth"),
    ('grey', 'normal_t0'):
        (lambda: UNet(downsample=6, in_channels=1, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/grey2normal.pth"),

    ('sobel_edges', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, in_channels=1, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/edge2depth.pth"),
    ('sobel_edges', 'depth_zbuffer_t0'):
        (lambda: UNet(downsample=6, in_channels=1, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/edge2depth.pth"),

    ('wav', 'depth_zbuffer'):
        (lambda: UNet(downsample=5, in_channels=12, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/wavelet2depth.pth"),
    ('wav', 'depth_zbuffer_t0'):
        (lambda: UNet(downsample=5, in_channels=12, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/wavelet2depth.pth"),

    ('grey', 'depth_zbuffer'):
        (lambda: UNet(in_channels=1, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/grey2depth.pth"),
    ('grey', 'depth_zbuffer_t0'):
        (lambda: UNet(in_channels=1, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/wo_cons/grey2depth.pth"),


    ('wav', 'normal'):
        (lambda: UNet(downsample=6, in_channels=12, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/wavelet2normal.pth"),
    ('wav', 'normal_t0'):
        (lambda: UNet(downsample=6, in_channels=12, out_channels=6).cuda(), f"{MODELS_DIR}/wo_cons/wavelet2normal.pth"),


    # network merging models
    ('stackedr', 'reshading'):
            (lambda: UNet_w(in_channels=16, downsample=2, out_channels=8).cuda(), None),
    ('stackedr', 'depth_zbuffer'):
            (lambda: UNet_w(in_channels=16, downsample=2, out_channels=8).cuda(), None),
    ('stackedr', 'normal'):
            (lambda: UNet_w(in_channels=48, downsample=3, out_channels=8).cuda(), None),

    #multi-domain baseline models
    # ('stackedr', 'reshading'):
    #     (lambda: UNet(in_channels=28, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/reshade_multiview.pth"),
    # ('stackedr', 'depth_zbuffer'):
    #     (lambda: UNet(in_channels=28, downsample=6, out_channels=2).cuda(), f"{MODELS_DIR}/depth_multiview.pth"),
    # ('stackedr', 'normal'):
    #     (lambda: UNet(in_channels=28, downsample=6, out_channels=6).cuda(), f"{MODELS_DIR}/normal_multiview.pth"),
    
    #multi-task baseline model
    ('rgb', 'depthnormalreshade'):
        (lambda: UNet(in_channels=3, downsample=6, out_channels=10), f"{MODELS_DIR}/multitask.pth"),


}

class Transfer(nn.Module):

    def __init__(self, src_task, dest_task,
        checkpoint=True, name=None, model_type=None, path=None,
        pretrained=True, finetuned=False
    ):
        super().__init__()
        if isinstance(src_task, str) and isinstance(dest_task, str):
            src_task, dest_task = get_task(src_task), get_task(dest_task)

        self.src_task, self.dest_task, self.checkpoint = src_task, dest_task, checkpoint
        self.name = name or f"{src_task.name}2{dest_task.name}"
        saved_type, saved_path = None, None
        if model_type is None and path is None:
            saved_type, saved_path = pretrained_transfers.get((src_task.name, dest_task.name), (None, None))

        self.model_type, self.path = model_type or saved_type, path or saved_path
        self.model = None

        if finetuned:
            path = f"{MODELS_DIR}/ft_perceptual/{src_task.name}2{dest_task.name}.pth"
            if os.path.exists(path):
                self.model_type, self.path = saved_type or (lambda: get_model(src_task, dest_task)), path
                print ("Using finetuned: ", path)
                return

        if self.model_type is None:

            if src_task.kind == dest_task.kind and src_task.resize != dest_task.resize:

                class Module(TrainableModel):

                    def __init__(self):
                        super().__init__()

                    def forward(self, x):
                        return resize(x, val=dest_task.resize)

                self.model_type = lambda: Module()
                self.path = None

            path = f"{MODELS_DIR}/{src_task.name}2{dest_task.name}.pth"
            if src_task.name == "keypoints2d" or dest_task.name == "keypoints2d":
                path = f"{MODELS_DIR}/{src_task.name}2{dest_task.name}_new.pth"
            if os.path.exists(path):
                self.model_type, self.path = lambda: get_model(src_task, dest_task), path

        if not pretrained:
            print ("Not using pretrained [heavily discouraged]")
            self.path = None

    def load_model(self):
        if self.model is None:
            if self.path is not None:
                self.model = DataParallelModel.load(self.model_type().to(DEVICE), self.path)
                # if optimizer:
                #     self.model.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
            else:
                self.model = self.model_type()
                if isinstance(self.model, nn.Module):
                    self.model = DataParallelModel(self.model)
        return self.model

    def __call__(self, x):
        self.load_model()
        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        preds.task = self.dest_task
        return preds

    def __repr__(self):
        return self.name or str(self.src_task) + " -> " + str(self.dest_task)


class RealityTransfer(Transfer):

    def __init__(self, src_task, dest_task):
        super().__init__(src_task, dest_task, model_type=lambda: None)

    def load_model(self, optimizer=True):
        pass

    def __call__(self, x):
        assert (isinstance(self.src_task, RealityTask))
        return self.src_task.task_data[self.dest_task].to(DEVICE)


class FineTunedTransfer(Transfer):

    def __init__(self, transfer):
        super().__init__(transfer.src_task, transfer.dest_task, checkpoint=transfer.checkpoint, name=transfer.name)
        self.cached_models = {}

    def load_model(self, parents=[]):

        model_path = get_finetuned_model_path(parents + [self])

        if model_path not in self.cached_models:
            if not os.path.exists(model_path):
                print(f"{model_path} not found, loading pretrained")
                self.cached_models[model_path] = super().load_model()
            else:
                print(f"{model_path} found, loading finetuned")
                self.cached_models[model_path] = DataParallelModel.load(self.model_type().cuda(), model_path)
                print(f"")
        self.model = self.cached_models[model_path]
        return self.model

    def __call__(self, x):

        if not hasattr(x, "parents"):
            x.parents = []

        self.load_model(parents=x.parents)
        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        preds.parents = x.parents + ([self])
        return preds



functional_transfers = (
    Transfer('normal', 'principal_curvature', name='f'),
    Transfer('principal_curvature', 'normal', name='F'),

    Transfer('normal', 'depth_zbuffer', name='g'),
    Transfer('depth_zbuffer', 'normal', name='G'),

    Transfer('normal', 'sobel_edges', name='s'),
    Transfer('sobel_edges', 'normal', name='S'),

    Transfer('principal_curvature', 'sobel_edges', name='CE'),
    Transfer('sobel_edges', 'principal_curvature', name='EC'),

    Transfer('depth_zbuffer', 'sobel_edges', name='DE'),
    Transfer('sobel_edges', 'depth_zbuffer', name='ED'),

    Transfer('principal_curvature', 'depth_zbuffer', name='h'),
    Transfer('depth_zbuffer', 'principal_curvature', name='H'),

    Transfer('rgb', 'normal', name='n'),
    Transfer('rgb', 'normal', name='npstep',
        model_type=lambda: UNetOld(),
        path=f"{MODELS_DIR}/unet_percepstep_0.1.pth",
    ),
    Transfer('rgb', 'principal_curvature', name='RC'),
    Transfer('rgb', 'keypoints2d', name='k'),
    Transfer('rgb', 'sobel_edges', name='a'),
    Transfer('rgb', 'reshading', name='r'),
    Transfer('rgb', 'depth_zbuffer', name='d'),

    Transfer('keypoints2d', 'principal_curvature', name='KC'),

    Transfer('keypoints3d', 'principal_curvature', name='k3C'),
    Transfer('principal_curvature', 'keypoints3d', name='Ck3'),

    Transfer('normal', 'reshading', name='nr'),
    Transfer('reshading', 'normal', name='rn'),

    Transfer('keypoints3d', 'normal', name='k3N'),
    Transfer('normal', 'keypoints3d', name='Nk3'),

    Transfer('keypoints2d', 'normal', name='k2N'),
    Transfer('normal', 'keypoints2d', name='Nk2'),

    Transfer('sobel_edges', 'reshading', name='Er'),
)

finetuned_transfers = [FineTunedTransfer(transfer) for transfer in functional_transfers]
TRANSFER_MAP = {t.name:t for t in functional_transfers}
functional_transfers = namedtuple('functional_transfers', TRANSFER_MAP.keys())(**TRANSFER_MAP)

def get_transfer_name(transfer):
    for t in functional_transfers:
        if transfer.src_task == t.src_task and transfer.dest_task == t.dest_task:
            return t.name
    return transfer.name

(f, F, g, G, s, S, CE, EC, DE, ED, h, H, n, npstep, RC, k, a, r, d, KC, k3C, Ck3, nr, rn, k3N, Nk3, Er, k2N, N2k) = functional_transfers

if __name__ == "__main__":
    y = g(F(f(x)))
    print (y.shape)






