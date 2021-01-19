import os, sys, math, random, itertools
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from task_configs import tasks, get_task, ImageTask
from transfers import functional_transfers, finetuned_transfers, get_transfer_name, Transfer
from datasets import TaskDataset, load_train_val

from matplotlib.cm import get_cmap


import IPython

def get_energy_loss(
    config="", mode="standard",
    pretrained=True, finetuned=True, **kwargs,
):
    """ Loads energy loss from config dict. """
    if isinstance(mode, str):
        mode = {
            "standard": EnergyLoss,
        }[mode]
    return mode(**energy_configs[config],
        pretrained=pretrained, finetuned=finetuned, **kwargs
    )


energy_configs = {


    ### normal configs

    "consistency_rgbnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "reshading": [tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "f(y^)": [tasks.normal, tasks.reshading],
            "g(y^)": [tasks.normal, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [[tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.imagenet],],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_reshade": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },

            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(y^)",
                    "g(n(x))",                    
                ]
            ),
        },
    },

    "consistency_wavnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.wav, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.wav, tasks.normal, tasks.reshading],
            "f(y^)": [tasks.normal, tasks.reshading],
            "g(y^)": [tasks.normal, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.wav, tasks.normal, tasks.principal_curvature],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.wav, tasks.normal, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.wav, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [[tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_reshade": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_greynormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "grey(x)": [tasks.rgb, tasks.grey],
            "n(x)": [tasks.rgb, tasks.grey, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.grey, tasks.normal, tasks.reshading],
            "f(y^)": [tasks.normal, tasks.reshading],
            "g(y^)": [tasks.normal, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.grey, tasks.normal, tasks.principal_curvature],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.grey, tasks.normal, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.grey, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [[tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_reshade": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "grey(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_embossnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "emboss(x)": [tasks.rgb, tasks.emboss4d],
            "n(x)": [tasks.rgb, tasks.emboss4d, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.emboss4d, tasks.normal, tasks.reshading],
            "f(y^)": [tasks.normal, tasks.reshading],
            "g(y^)": [tasks.normal, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.emboss4d, tasks.normal, tasks.principal_curvature],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.emboss4d, tasks.normal, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.emboss4d, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [[tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_reshade": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "emboss(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_edgenormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "sobel(x)": [tasks.rgb, tasks.sobel_edges],
            "n(x)": [tasks.rgb, tasks.sobel_edges, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal, tasks.reshading],
            "f(y^)": [tasks.normal, tasks.reshading],
            "g(y^)": [tasks.normal, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal, tasks.principal_curvature],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [[tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_reshade": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "sobel(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },
    
    "consistency_sharpnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "sharp(x)": [tasks.rgb, tasks.sharp],
            "n(x)": [tasks.rgb, tasks.sharp, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.sharp, tasks.normal, tasks.reshading],
            "f(y^)": [tasks.normal, tasks.reshading],
            "g(y^)": [tasks.normal, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.sharp, tasks.normal, tasks.principal_curvature],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.sharp, tasks.normal, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.sharp, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [[tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_reshade": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "sharp(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },
    
    "consistency_gaussnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "gauss(x)": [tasks.rgb, tasks.gauss],
            "n(x)": [tasks.rgb, tasks.gauss, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.gauss, tasks.normal, tasks.reshading],
            "f(y^)": [tasks.normal, tasks.reshading],
            "g(y^)": [tasks.normal, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.gauss, tasks.normal, tasks.principal_curvature],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.gauss, tasks.normal, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.gauss, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [[tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_reshade": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "gauss(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_laplacenormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "laplace(x)": [tasks.rgb, tasks.laplace_edges],
            "n(x)": [tasks.rgb, tasks.laplace_edges, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.normal, tasks.reshading],
            "f(y^)": [tasks.normal, tasks.reshading],
            "g(y^)": [tasks.normal, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.normal, tasks.principal_curvature],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.normal, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [[tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_reshade": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "laplace(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    ### depth config

    "consistency_rgbdepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.normal],
            "f(y^)": [tasks.depth_zbuffer, tasks.normal],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.principal_curvature],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.edge_occlusion],
            #"imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            #"imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [[tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion]
            #[tasks.depth_zbuffer, tasks.imagenet]
            ],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_reshade": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },

            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },

            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },

            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },

            #"percep_imagenet_percep": {
            #    ("train", "val"): [
            #        ("imagenet(n(x))", "imagenet(y^)"),
            #    ],
            #},
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "s(n(x))",
                    "s(y^)",
                    "g(n(x))",
                    "g(y^)",
                    "Nk3(n(x))",
                    "Nk3(y^)",
                    "Nk2(n(x))",
                    "Nk2(y^)",
                    "nEO(n(x))",
                    "nEO(y^)",
                ]
            ),
        },
    },
    
    "consistency_edgedepth": {
        "paths": {
            "x": [tasks.rgb],
            "edge(x)": [tasks.rgb, tasks.sobel_edges],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer, tasks.normal],
            "f(y^)": [tasks.depth_zbuffer, tasks.normal],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer, tasks.principal_curvature],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer, tasks.edge_occlusion],
            #"imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            #"imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [[tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion]
            #[tasks.depth_zbuffer, tasks.imagenet]
            ],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_reshade": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },

            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },

            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },

            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },

            #"percep_imagenet_percep": {
            #    ("train", "val"): [
            #        ("imagenet(n(x))", "imagenet(y^)"),
            #    ],
            #},
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "edge(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "s(n(x))",
                    "s(y^)",
                    "g(n(x))",
                    "g(y^)",
                    "Nk3(n(x))",
                    "Nk3(y^)",
                    "Nk2(n(x))",
                    "Nk2(y^)",
                    "nEO(n(x))",
                    "nEO(y^)",
                ]
            ),
        },
    },

    "consistency_embossdepth": {
        "paths": {
            "x": [tasks.rgb],
            "emboss4d(x)": [tasks.rgb, tasks.emboss4d],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer, tasks.normal],
            "f(y^)": [tasks.depth_zbuffer, tasks.normal],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer, tasks.principal_curvature],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer, tasks.edge_occlusion],
            #"imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            #"imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [[tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion]
            #[tasks.depth_zbuffer, tasks.imagenet]
            ],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_reshade": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },

            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },

            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },

            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },

            #"percep_imagenet_percep": {
            #    ("train", "val"): [
            #        ("imagenet(n(x))", "imagenet(y^)"),
            #    ],
            #},
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "emboss4d(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "s(n(x))",
                    "s(y^)",
                    "g(n(x))",
                    "g(y^)",
                    "Nk3(n(x))",
                    "Nk3(y^)",
                    "Nk2(n(x))",
                    "Nk2(y^)",
                    "nEO(n(x))",
                    "nEO(y^)",
                ]
            ),
        },
    },
    
    "consistency_gaussdepth": {
        "paths": {
            "x": [tasks.rgb],
            "gauss(x)": [tasks.rgb, tasks.gauss],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer, tasks.normal],
            "f(y^)": [tasks.depth_zbuffer, tasks.normal],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer, tasks.principal_curvature],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer, tasks.edge_occlusion],
            #"imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            #"imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [[tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion]
            #[tasks.depth_zbuffer, tasks.imagenet]
            ],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_reshade": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },

            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },

            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },

            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },

            #"percep_imagenet_percep": {
            #    ("train", "val"): [
            #        ("imagenet(n(x))", "imagenet(y^)"),
            #    ],
            #},
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "gauss(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "s(n(x))",
                    "s(y^)",
                    "g(n(x))",
                    "g(y^)",
                    "Nk3(n(x))",
                    "Nk3(y^)",
                    "Nk2(n(x))",
                    "Nk2(y^)",
                    "nEO(n(x))",
                    "nEO(y^)",
                ]
            ),
        },
    },
    
    "consistency_sharpdepth": {
        "paths": {
            "x": [tasks.rgb],
            "sharp(x)": [tasks.rgb, tasks.sharp],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer, tasks.normal],
            "f(y^)": [tasks.depth_zbuffer, tasks.normal],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer, tasks.principal_curvature],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer, tasks.edge_occlusion],
            #"imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            #"imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [[tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion]
            #[tasks.depth_zbuffer, tasks.imagenet]
            ],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_reshade": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },

            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },

            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },

            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },

            #"percep_imagenet_percep": {
            #    ("train", "val"): [
            #        ("imagenet(n(x))", "imagenet(y^)"),
            #    ],
            #},
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "sharp(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "s(n(x))",
                    "s(y^)",
                    "g(n(x))",
                    "g(y^)",
                    "Nk3(n(x))",
                    "Nk3(y^)",
                    "Nk2(n(x))",
                    "Nk2(y^)",
                    "nEO(n(x))",
                    "nEO(y^)",
                ]
            ),
        },
    },
    
    "consistency_laplacedepth": {
        "paths": {
            "x": [tasks.rgb],
            "laplace(x)": [tasks.rgb, tasks.laplace_edges],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer, tasks.normal],
            "f(y^)": [tasks.depth_zbuffer, tasks.normal],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer, tasks.principal_curvature],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer, tasks.edge_occlusion],
            #"imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            #"imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [[tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion]
            #[tasks.depth_zbuffer, tasks.imagenet]
            ],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_reshade": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },

            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },

            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },

            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },

            #"percep_imagenet_percep": {
            #    ("train", "val"): [
            #        ("imagenet(n(x))", "imagenet(y^)"),
            #    ],
            #},
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "laplace(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "s(n(x))",
                    "s(y^)",
                    "g(n(x))",
                    "g(y^)",
                    "Nk3(n(x))",
                    "Nk3(y^)",
                    "Nk2(n(x))",
                    "Nk2(y^)",
                    "nEO(n(x))",
                    "nEO(y^)",
                ]
            ),
        },
    },
    
    "consistency_greydepth": {
        "paths": {
            "x": [tasks.rgb],
            "grey(x)": [tasks.rgb, tasks.grey],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.grey, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer, tasks.normal],
            "f(y^)": [tasks.depth_zbuffer, tasks.normal],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer, tasks.principal_curvature],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer, tasks.edge_occlusion],
            #"imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            #"imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [[tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion]
            #[tasks.depth_zbuffer, tasks.imagenet]
            ],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_reshade": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },

            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },

            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },

            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },

            #"percep_imagenet_percep": {
            #    ("train", "val"): [
            #        ("imagenet(n(x))", "imagenet(y^)"),
            #    ],
            #},
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "grey(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "s(n(x))",
                    "s(y^)",
                    "g(n(x))",
                    "g(y^)",
                    "Nk3(n(x))",
                    "Nk3(y^)",
                    "Nk2(n(x))",
                    "Nk2(y^)",
                    "nEO(n(x))",
                    "nEO(y^)",
                ]
            ),
        },
    },

    "consistency_wavdepth": {
        "paths": {
            "x": [tasks.rgb],
            "wav(x)": [tasks.rgb, tasks.wav],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.wav, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer, tasks.normal],
            "f(y^)": [tasks.depth_zbuffer, tasks.normal],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer, tasks.principal_curvature],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer, tasks.edge_occlusion],
            #"imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            #"imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [[tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion]
            #[tasks.depth_zbuffer, tasks.imagenet]
            ],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_reshade": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },

            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },

            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },

            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },

            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },

            #"percep_imagenet_percep": {
            #    ("train", "val"): [
            #        ("imagenet(n(x))", "imagenet(y^)"),
            #    ],
            #},
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "s(n(x))",
                    "s(y^)",
                    "g(n(x))",
                    "g(y^)",
                    "Nk3(n(x))",
                    "Nk3(y^)",
                    "Nk2(n(x))",
                    "Nk2(y^)",
                    "nEO(n(x))",
                    "nEO(y^)",
                ]
            ),
        },
    },

    ### reshade config

    "consistency_rgbreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "n(x)": [tasks.rgb, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.reshading, tasks.normal],
            "f(y^)": [tasks.reshading, tasks.normal],
            "g(y^)": [tasks.reshading, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.reshading, tasks.principal_curvature],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.reshading, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [[tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_edgereshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "sobel(x)": [tasks.rgb, tasks.sobel_edges],
            "n(x)": [tasks.rgb, tasks.sobel_edges, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.reshading, tasks.normal],
            "f(y^)": [tasks.reshading, tasks.normal],
            "g(y^)": [tasks.reshading, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.reshading, tasks.principal_curvature],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.reshading, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [[tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "sobel(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_greyreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "grey(x)": [tasks.rgb, tasks.grey],
            "n(x)": [tasks.rgb, tasks.grey, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.grey, tasks.reshading, tasks.normal],
            "f(y^)": [tasks.reshading, tasks.normal],
            "g(y^)": [tasks.reshading, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.grey, tasks.reshading, tasks.principal_curvature],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.grey, tasks.reshading, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.grey, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [[tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "grey(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_embossreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "emboss(x)": [tasks.rgb, tasks.emboss],
            "n(x)": [tasks.rgb, tasks.emboss, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.emboss, tasks.reshading, tasks.normal],
            "f(y^)": [tasks.reshading, tasks.normal],
            "g(y^)": [tasks.reshading, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.emboss, tasks.reshading, tasks.principal_curvature],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.emboss, tasks.reshading, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.emboss, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [[tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.imagenet]],
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "emboss(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_wavreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "n(x)": [tasks.rgb, tasks.wav, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.wav, tasks.reshading, tasks.normal],
            "f(y^)": [tasks.reshading, tasks.normal],
            "g(y^)": [tasks.reshading, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.wav, tasks.reshading, tasks.principal_curvature],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.wav, tasks.reshading, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.wav, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [[tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },

    "consistency_gaussreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "gauss(x)": [tasks.rgb, tasks.gauss],
            "n(x)": [tasks.rgb, tasks.gauss, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.gauss, tasks.reshading, tasks.normal],
            "f(y^)": [tasks.reshading, tasks.normal],
            "g(y^)": [tasks.reshading, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.gauss, tasks.reshading, tasks.principal_curvature],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.gauss, tasks.reshading, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.gauss, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [[tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "gauss(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },
    
    "consistency_laplacereshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "laplace(x)": [tasks.rgb, tasks.laplace_edges],
            "n(x)": [tasks.rgb, tasks.laplace_edges, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.reshading, tasks.normal],
            "f(y^)": [tasks.reshading, tasks.normal],
            "g(y^)": [tasks.reshading, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.reshading, tasks.principal_curvature],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.reshading, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [[tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },   
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "laplace(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },
    
    "consistency_sharpreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "sharp(x)": [tasks.rgb, tasks.sharp],
            "n(x)": [tasks.rgb, tasks.sharp, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.sharp, tasks.reshading, tasks.normal],
            "f(y^)": [tasks.reshading, tasks.normal],
            "g(y^)": [tasks.reshading, tasks.principal_curvature],
            "g(n(x))": [tasks.rgb, tasks.sharp, tasks.reshading, tasks.principal_curvature],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.sharp, tasks.reshading, tasks.depth_zbuffer],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.sharp, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [[tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.imagenet]],
            
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },

            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            
            "percep_curv": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "sharp(x)",
                    "n(x)",
                    "f(n(x))",
                    "f(y^)",
                    "nr(n(x))",
                    "nr(y^)",
                    "g(n(x))",
                    "g(y^)",
                ]
            ),
        },
    },
    

}



def coeff_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1


class EnergyLoss(object):

    def __init__(self, paths, losses, plots,
        pretrained=True, finetuned=False, freeze_list=[]
    ):

        self.paths, self.losses, self.plots = paths, losses, plots
        self.freeze_list = [str((path[0].name, path[1].name)) for path in freeze_list]
        self.metrics = {}

        self.tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    self.tasks += self.paths[path1] + self.paths[path2]

        for name, config in self.plots.items():
            for path in config["paths"]:
                self.tasks += self.paths[path]
        self.tasks = list(set(self.tasks))

    def compute_paths(self, graph, reality=None, paths=None):
        path_cache = {}
        paths = paths or self.paths
        path_values = {
            name: graph.sample_path(path,
                reality=reality, use_cache=True, cache=path_cache, name=name
            ) for name, path in paths.items()
        }
        del path_cache
        return {k: v for k, v in path_values.items() if v is not None}

    def get_tasks(self, reality):
        tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                if reality in realities:
                    for path1, path2 in losses:
                        tasks += [self.paths[path1][0], self.paths[path2][0]]

        for name, config in self.plots.items():
            if reality in config["realities"]:
                for path in config["paths"]:
                    tasks += [self.paths[path][0]]

        return list(set(tasks))

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, batch_mean=True, use_l1=False):
        #pdb.set_trace()
        loss = {}
        for reality in realities:
            loss_dict = {}
            losses = []
            all_loss_types = set()
            for loss_type, loss_item in self.losses.items():
                all_loss_types.add(loss_type)
                loss_dict[loss_type] = []
                for realities_l, data in loss_item.items():
                    if reality.name in realities_l:
                        loss_dict[loss_type] += data
                        if loss_types is not None and loss_type in loss_types:
                            losses += data

            path_values = self.compute_paths(graph,
                paths={
                    path: self.paths[path] for path in \
                    set(path for paths in losses for path in paths)
                    },
                reality=reality)

            if reality.name not in self.metrics:
                self.metrics[reality.name] = defaultdict(list)

            mask = ImageTask.build_mask(path_values["y^"][:,:3], val=self.paths['y^'][0].mask_val).float()

            for loss_type, losses in sorted(loss_dict.items()):
                if loss_type not in (loss_types or all_loss_types):
                    continue
                if loss_type not in loss:
                    loss[loss_type] = 0
                for path1, path2 in losses:
                    # print(loss_type,path1,path2)
                    output_task = self.paths[path1][-1]
                    if loss_type not in loss:
                        loss[loss_type] = 0
                    for path1, path2 in losses:
                        output_task = self.paths[path1][-1]
                        compute_mask = 'imagenet(n(x))' != path1
                        if (loss_type=='nll'):
                            path_loss_nll, _ = output_task.nll(path_values[path1], path_values[path2], batch_mean=batch_mean, compute_mask=True, mask=mask)
                            loss[loss_type] += path_loss_nll
                            self.metrics[reality.name]["nll : "+path1 + " -> " + path2] += [path_loss_nll.mean().detach().cpu()]
                        if path1=='n(x)':
                            nchannels = path_values['n(x)'].size(1)//2
                        else:
                            nchannels = path_values[path1].size(1)
                        path_loss, _ = output_task.norm(path_values[path1][:,:nchannels], path_values[path2][:,:nchannels], batch_mean=batch_mean, compute_mask=compute_mask, compute_mse=False, mask=mask)
                        if 'percep' in loss_type: loss[loss_type] += path_loss
                        self.metrics[reality.name]["mae : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]
                        path_loss, _ = output_task.norm(path_values[path1][:,:nchannels], path_values[path2][:,:nchannels], batch_mean=batch_mean, compute_mask=compute_mask, compute_mse=True, mask=mask)
                        self.metrics[reality.name]["mse : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

        return loss

    def logger_hooks(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    if (loss_type=='nll'):
                        name = "nll : "+path1 + " -> " + path2
                        name_to_realities[name] += list(realities)
                    name = "mae : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    name = "mse : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):
                names = [f"{reality}_{name}" for reality in realities]
                if not all(x in data for x in names):
                    return
                data = np.stack([data[x] for x in names], axis=1)
                logger.plot(data, name, opts={"legend": names})

            logger.add_hook(partial(jointplot, name=name, realities=realities), feature=f"{realities[-1]}_{name}", freq=1)


    def logger_update(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    if (loss_type=='nll'):
                        name = "nll : "+path1 + " -> " + path2
                        name_to_realities[name] += list(realities)
                    name = "mae : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    name = "mse : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            for reality in realities:
                # IPython.embed()
                if reality not in self.metrics: continue
                if name not in self.metrics[reality]: continue
                if len(self.metrics[reality][name]) == 0: continue

                logger.update(
                    f"{reality}_{name}",
                    torch.mean(torch.stack(self.metrics[reality][name])),
                )
        self.metrics = {}

    def plot_paths(self, graph, logger, realities=[], plot_names=None, epochs=0, tr_step=0,prefix=""):


        SQRT2 = math.sqrt(2)
        path_values = {}
        realities_map = {reality.name: reality for reality in realities}
        for name, config in (plot_names or self.plots.items()):
            paths = config["paths"]

            cmap = get_cmap("jet")
            realities = config["realities"]
            ind = np.diag_indices(3)
            for reality in realities:
                with torch.no_grad():

                    path_values[reality] = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])

                    if reality is 'test': #compute error map
                        mask_task = self.paths["y^"][-1]
                        mask = ImageTask.build_mask(path_values[reality]["y^"], val=mask_task.mask_val)
                        errors = ((path_values[reality]["y^"][:,:1]-path_values[reality]["n(x)"][:,:1])**2).mean(dim=1, keepdim=True)
                        errors = (3*errors/(mask_task.variance)).clamp(min=0, max=1)
                        log_errors = torch.log(errors + 1)
                        log_errors = log_errors / log_errors.max()
                        log_errors = torch.tensor(cmap(log_errors.cpu()))[:, 0].permute((0, 3, 1, 2)).float()[:, 0:3]
                        log_errors = log_errors.clamp(min=0, max=1).to(DEVICE)
                        log_errors[~mask.expand_as(log_errors)] = 0.505
                        path_values[reality]['error']= log_errors

                    for p in  self.plots['']['paths']:
                        if p in ['x','y^']: continue
                        if ('y^' in p or 'depth' in p or 'normal' in p) and ('ood' in reality): continue
                        if (p=='n(x)'): 
                            nchannels = path_values[reality][p].size(1) // 2
                            path_values[reality][f'{p}_mu'] = path_values[reality][p][:,:nchannels]
                            path_values[reality][f'{p}_sigma'] = path_values[reality][p][:,nchannels:].exp()*SQRT2
                            del path_values[reality][p]
                        else: # stupid hack so n(x) and prob perceps are plotted first
                            tmp = path_values[reality][p]
                            del path_values[reality][p]
                            path_values[reality][p] = tmp

                    path_values[reality] = {k:v.clamp(min=0,max=1).cpu() for k,v in path_values[reality].items()}

        # more processing
        def reshape_img_to_rows(x_):
            downsample = lambda x: F.interpolate(x.unsqueeze(0),scale_factor=0.5,mode='bilinear').squeeze(0)
            x_list = [downsample(x_[i]) for i in range(x_.size(0))]
            x=torch.cat(x_list,dim=-1)
            return x


        all_images = {}
        for reality in realities:
            all_imgs_reality = []
            plot_name = ''
            for k in path_values[reality].keys():
                plot_name += k+'_'
                if path_values[reality][k].size(1)>3: path_values[reality][k]=path_values[reality][k][:,:3]
                img_row = reshape_img_to_rows(path_values[reality][k])
                if img_row.size(0) == 1: img_row = img_row.repeat(3,1,1)
                all_imgs_reality.append(img_row)
            plot_name = plot_name[:-1]
            all_images[reality] = torch.cat(all_imgs_reality,dim=-2)

        return all_images


    def __repr__(self):
        return str(self.losses)


class EnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 3)
        self.random_select = kwargs.pop('random_select', False)
        self.running_stats = {}
        self.target_task = kwargs['paths']['y^'][0].name
        self.mid_domain = kwargs['paths']['n(x)'][1].name if len(kwargs['paths']['n(x)'])==3 else 'rgb'

        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
        print (self.percep_losses)
        self.chosen_losses = self.percep_losses


    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, compute_grad_ratio=False):

        loss_types = ["nll"] + [("percep_" + loss) for loss in self.percep_losses]
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, batch_mean=False)

        chosen_percep_mse_losses = [k for k in loss_dict.keys() if 'direct' not in k]
        percep_mse_coeffs = dict.fromkeys(chosen_percep_mse_losses, 1.0)
        ########### to compute loss coefficients #############
        if compute_grad_ratio:
            percep_mse_gradnorms = dict.fromkeys(chosen_percep_mse_losses, 1.0)
            for loss_name in chosen_percep_mse_losses:
                graph.optimizer.zero_grad()
                graph.zero_grad()
                loss_dict[loss_name].mean().backward(retain_graph=True) # retain_graph=True so that backprop can be done again
                target_weights=list(graph.edge_map[f"('{self.mid_domain}', '{self.target_task}')"].model.parameters())
                percep_mse_gradnorms[loss_name] = sum([l.grad.abs().sum().item() for l in target_weights])/sum([l.numel() for l in target_weights])
                graph.optimizer.zero_grad()
                graph.zero_grad()
                del target_weights
            total_gradnorms = sum(percep_mse_gradnorms.values())
            n_losses = len(chosen_percep_mse_losses)
            for loss_name, val in percep_mse_coeffs.items():
                percep_mse_coeffs[loss_name] = (total_gradnorms-percep_mse_gradnorms[loss_name])/((n_losses-1)*total_gradnorms)
            percep_mse_coeffs["nll"] *= (n_losses-1)

        for key in self.chosen_losses:
            loss_dict[f"percep_{key}"] = loss_dict[f"percep_{key}"].mean() * percep_mse_coeffs[f"percep_{key}"]

        percep_mse_gradnorms = {k+'_grad':v for k,v in percep_mse_gradnorms.items()} if compute_grad_ratio else None
        percep_mse_coeffs[f"nll"] = percep_mse_coeffs[f"nll"] / 10.0

        loss_dict["nll"] = loss_dict["nll"].mean() * percep_mse_coeffs["nll"]

        return loss_dict, percep_mse_coeffs, percep_mse_gradnorms

    def logger_update(self, logger):
        super().logger_update(logger)


