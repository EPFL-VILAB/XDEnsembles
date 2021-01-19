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


    # reshade configs

    "trainsig_rgbreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "f(g(x))": [tasks.rgb, tasks.reshading],
            "f0(g(x))": [tasks.rgb, tasks.reshading_t0],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [[tasks.rgb, tasks.reshading_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_edgereshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "sobel(x)": [tasks.rgb, tasks.sobel_edges],
            "f(g(x))": [tasks.rgb, tasks.sobel_edges, tasks.reshading],
            "f0(g(x))": [tasks.rgb, tasks.sobel_edges, tasks.reshading_t0],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [[tasks.sobel_edges, tasks.reshading_t0],
                        [tasks.rgb, tasks.sobel_edges]],
        "losses": {
            "main": {
                ("train_undist", "val_undist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_undist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_undist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "sobel(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_embossreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "emboss(x)": [tasks.rgb, tasks.emboss4d],
            "f(g(x))": [tasks.rgb, tasks.emboss4d, tasks.reshading],
            "f0(g(x))": [tasks.rgb, tasks.emboss4d, tasks.reshading_t0],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [[tasks.emboss4d, tasks.reshading_t0],
                        [tasks.rgb, tasks.emboss4d]],
        "losses": {
            "main": {
                ("train_undist", "val_undist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_undist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_undist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "emboss(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_greyreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "grey(x)": [tasks.rgb, tasks.grey],
            "f(g(x))": [tasks.rgb, tasks.grey, tasks.reshading],
            "f0(g(x))": [tasks.rgb, tasks.grey, tasks.reshading_t0],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [[tasks.grey, tasks.reshading_t0],
                        [tasks.rgb, tasks.grey]],
        "losses": {
            "main": {
                ("train_undist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "grey(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_wavreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "wav(x)": [tasks.rgb, tasks.wav],
            "f(g(x))": [tasks.rgb, tasks.wav, tasks.reshading],
            "f0(g(x))": [tasks.rgb, tasks.wav, tasks.reshading_t0],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [[tasks.wav, tasks.reshading_t0],
                        [tasks.rgb, tasks.wav]],
        "losses": {
            "main": {
                ("train_undist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    # "wav(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_sharpreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "sharp(x)": [tasks.rgb, tasks.sharp],
            "f(g(x))": [tasks.rgb, tasks.sharp, tasks.reshading],
            "f0(g(x))": [tasks.rgb, tasks.sharp, tasks.reshading_t0],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [[tasks.sharp, tasks.reshading_t0],
                        [tasks.rgb, tasks.sharp]],
        "losses": {
            "main": {
                ("train_undist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "sharp(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_lapedgereshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "lap(x)": [tasks.rgb, tasks.laplace_edges],
            "f(g(x))": [tasks.rgb, tasks.laplace_edges, tasks.reshading],
            "f0(g(x))": [tasks.rgb, tasks.laplace_edges, tasks.reshading_t0],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [[tasks.laplace_edges, tasks.reshading_t0],
                        [tasks.rgb, tasks.laplace_edges]],
        "losses": {
            "main": {
                ("train_undist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "lap(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_gblurreshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "blur(x)": [tasks.rgb, tasks.gauss],
            "f(g(x))": [tasks.rgb, tasks.gauss, tasks.reshading],
            "f0(g(x))": [tasks.rgb, tasks.gauss, tasks.reshading_t0],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [[tasks.gauss, tasks.reshading_t0],
                        [tasks.rgb, tasks.gauss]],
        "losses": {
            "main": {
                ("train_undist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "blur(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },


    # depth configs

    "trainsig_rgbdepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "f(g(x))": [tasks.rgb, tasks.depth_zbuffer],
            "f0(g(x))": [tasks.rgb, tasks.depth_zbuffer_t0],
        },
        "freeze_list": [[tasks.rgb, tasks.depth_zbuffer_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_edgedepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "sobel(x)": [tasks.rgb, tasks.sobel_edges],
            "f(g(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer],
            "f0(g(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer_t0],
        },
        "freeze_list": [[tasks.sobel_edges, tasks.depth_zbuffer_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "sobel(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_embossdepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "emboss(x)": [tasks.rgb, tasks.emboss4d],
            "f(g(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer],
            "f0(g(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer_t0],
        },
        "freeze_list": [[tasks.emboss4d, tasks.depth_zbuffer_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "emboss(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_greydepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "grey(x)": [tasks.rgb, tasks.grey],
            "f(g(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer],
            "f0(g(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer_t0],
        },
        "freeze_list": [[tasks.grey, tasks.depth_zbuffer_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "grey(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_wavdepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            # "wav(x)": [tasks.rgb, tasks.wav],
            "f(g(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer],
            "f0(g(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer_t0],
        },
        "freeze_list": [[tasks.wav, tasks.depth_zbuffer_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    # "wav(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_sharpdepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "sharp(x)": [tasks.rgb, tasks.sharp],
            "f(g(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer],
            "f0(g(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer_t0],
        },
        "freeze_list": [[tasks.sharp, tasks.depth_zbuffer_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "sharp(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_lapedgedepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "lap(x)": [tasks.rgb, tasks.laplace_edges],
            "f(g(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer],
            "f0(g(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer_t0],
        },
        "freeze_list": [[tasks.laplace_edges, tasks.depth_zbuffer_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "lap(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_gblurdepth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "gauss(x)": [tasks.rgb, tasks.gauss],
            "f(g(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer],
            "f0(g(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer_t0],
        },
        "freeze_list": [[tasks.gauss, tasks.depth_zbuffer_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "gauss(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },


    # normal configs
    
    "trainsig_rgbnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "f(g(x))": [tasks.rgb, tasks.normal],
            "f0(g(x))": [tasks.rgb, tasks.normal_t0],
            "reshade": [tasks.normal],
        },
        "freeze_list": [[tasks.rgb, tasks.normal_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_edgenormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "sobel(x)": [tasks.rgb, tasks.sobel_edges],
            "f(g(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal],
            "f0(g(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal_t0],
            "reshade": [tasks.normal],
        },
        "freeze_list": [[tasks.sobel_edges, tasks.normal_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "sobel(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_greynormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "grey(x)": [tasks.rgb, tasks.grey],
            "f(g(x))": [tasks.rgb, tasks.grey, tasks.normal],
            "f0(g(x))": [tasks.rgb, tasks.grey, tasks.normal_t0],
            "reshade": [tasks.normal],
        },
        "freeze_list": [[tasks.grey, tasks.normal_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "grey(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_wavnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "wav(x)": [tasks.rgb, tasks.wav],
            "f(g(x))": [tasks.rgb, tasks.wav, tasks.normal],
            "f0(g(x))": [tasks.rgb, tasks.wav, tasks.normal_t0],
            "reshade": [tasks.normal],
        },
        "freeze_list": [[tasks.wav, tasks.normal_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    # "wav(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_embossnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "emboss(x)": [tasks.rgb, tasks.emboss4d],
            "f(g(x))": [tasks.rgb, tasks.emboss4d, tasks.normal],
            "f0(g(x))": [tasks.rgb, tasks.emboss4d, tasks.normal_t0],
            "reshade": [tasks.normal],
        },
        "freeze_list": [[tasks.emboss4d, tasks.normal_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "emboss(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_sharpnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "sharp(x)": [tasks.rgb, tasks.sharp],
            "f(g(x))": [tasks.rgb, tasks.sharp, tasks.normal],
            "f0(g(x))": [tasks.rgb, tasks.sharp, tasks.normal_t0],
            "reshade": [tasks.normal],
        },
        "freeze_list": [[tasks.sharp, tasks.normal_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "sharp(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_lapedgenormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "lap(x)": [tasks.rgb, tasks.laplace_edges],
            "f(g(x))": [tasks.rgb, tasks.laplace_edges, tasks.normal],
            "f0(g(x))": [tasks.rgb, tasks.laplace_edges, tasks.normal_t0],
        },
        "freeze_list": [[tasks.laplace_edges, tasks.normal_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "lap(x)",
                    "f(g(x))",
                    "f0(g(x))"
                ]
            ),
        },
    },

    "trainsig_gblurnormal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "blur(x)": [tasks.rgb, tasks.gauss],
            "f(g(x))": [tasks.rgb, tasks.gauss, tasks.normal],
            "f0(g(x))": [tasks.rgb, tasks.gauss, tasks.normal_t0],
        },
        "freeze_list": [[tasks.gauss, tasks.normal_t0]],
        "losses": {
            "main": {
                ("train_undist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
            "lwf": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "f0(g(x))"),
                ],
            },
            "sig": {
                ("train_dist", "val_ooddist", "val_dist", "val"): [
                    ("f(g(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug", "ood_syn"),
                paths=[
                    "x",
                    "y^",
                    "blur(x)",
                    "f(g(x))",
                    "f0(g(x))"
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
                reality=reality, use_cache=True, cache=path_cache,
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

            mask = None
            mask = self.paths['y^'][0].build_mask(path_values['y^'], val=self.paths['y^'][0].mask_val).float()
            for loss_type, losses in sorted(loss_dict.items()):
                if loss_type not in (loss_types or all_loss_types):
                    continue
                if loss_type not in loss:
                    loss[loss_type] = 0
                for path1, path2 in losses:
                    output_task = self.paths[path1][-1]
                    if loss_type not in loss:
                        loss[loss_type] = 0
                    for path1, path2 in losses:

                        output_task = self.paths[path1][-1]

                        if loss_type=='main':
                            path_loss_nll, _ = output_task.nll(path_values[path1], path_values[path2], batch_mean=batch_mean, compute_mask=True, mask=mask)
                            loss[loss_type] += path_loss_nll
                            self.metrics[reality.name][loss_type + "_nll : "+path1 + " -> " + path2] += [path_loss_nll.mean().detach().cpu()]
                        nchannels = path_values[path1].size(1) // 2
                        if loss_type in ['main','lwf']:
                            # standard mae loss
                            path_loss, _ = output_task.norm(path_values[path1][:,:nchannels], path_values[path2][:,:nchannels], batch_mean=batch_mean, compute_mask=True, compute_mse=False, mask=mask)
                        else:
                            # calibration loss: || sigma(x)-|mu(x)-y^| ||_1
                            abs_err = (path_values["f0(g(x))"][:,:nchannels]-path_values[path2][:,:nchannels]).abs()
                            path_loss, _ = output_task.norm(path_values[path1][:,nchannels:].exp(), abs_err, batch_mean=batch_mean, compute_mask=True, compute_mse=False, mask=mask)
                        if loss_type in ['lwf','sig']: loss[loss_type] += path_loss
                        self.metrics[reality.name][loss_type + "_mae : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

                        if loss_type in ['main','lwf']:
                            path_loss, _ = output_task.norm(path_values[path1][:,:nchannels], path_values[path2][:,:nchannels], batch_mean=batch_mean, compute_mask=True, compute_mse=True, mask=mask)
                        else:
                            abs_err = (path_values["f0(g(x))"][:,:nchannels]-path_values[path2][:,:nchannels]).abs()
                            path_loss, _ = output_task.norm(path_values[path1][:,nchannels:].exp(), abs_err, batch_mean=batch_mean, compute_mask=True, compute_mse=True, mask=mask)
                        self.metrics[reality.name][loss_type + "_mse : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

        return loss

    def logger_hooks(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    name = loss_type + "_nll : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    name = loss_type + "_mae : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    name = loss_type + "_mse : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):
                names = [f"{reality}|{name}" for reality in realities]
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
                    name = loss_type + "_nll : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    name = loss_type + "_mae : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    name = loss_type + "_mse : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            for reality in realities:
                # IPython.embed()
                if reality not in self.metrics: continue
                if name not in self.metrics[reality]: continue
                if len(self.metrics[reality][name]) == 0: continue

                logger.update(
                    f"{reality}|{name}",
                    torch.mean(torch.stack(self.metrics[reality][name])),
                )
        self.metrics = {}

    def plot_paths(self, graph, logger, realities=[], plot_names=None, epochs=0, tr_step=0,prefix=""):

        sqrt2 = math.sqrt(2)

        cmap = get_cmap("jet")
        path_values = {}
        realities_map = {reality.name: reality for reality in realities}
        for name, config in (plot_names or self.plots.items()):
            paths = config["paths"]

            realities = config["realities"]

            for reality in realities:
                with torch.no_grad():
                    # pdb.set_trace()
                    path_values[reality] = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])
                    if reality is 'test': #compute error map
                        mask_task = self.paths["y^"][-1]
                        mask = ImageTask.build_mask(path_values[reality]["y^"], val=mask_task.mask_val)
                        errors = ((path_values[reality]["y^"][:,:3]-path_values[reality]["f(g(x))"][:,:3])**2).mean(dim=1, keepdim=True)
                        errors = (3*errors/(mask_task.variance)).clamp(min=0, max=1)
                        log_errors = torch.log(errors + 1)
                        log_errors = log_errors / log_errors.max()
                        log_errors = torch.tensor(cmap(log_errors.cpu()))[:, 0].permute((0, 3, 1, 2)).float()[:, 0:3]
                        log_errors = log_errors.clamp(min=0, max=1).to(DEVICE)
                        log_errors[~mask.expand_as(log_errors)] = 0.505
                        path_values[reality]['error']= log_errors

                    nchannels = path_values[reality]['f(g(x))'].size(1) // 2
                    path_values[reality]['f(g(x))_m'] = path_values[reality]['f(g(x))'][:,:nchannels]
                    path_values[reality]['f(g(x))_s'] = path_values[reality]['f(g(x))'][:,nchannels:].exp()*sqrt2
                    path_values[reality]['f0(g(x))_m'] = path_values[reality]['f0(g(x))'][:,:nchannels]
                    path_values[reality]['f0(g(x))_s'] = path_values[reality]['f0(g(x))'][:,nchannels:].exp()*sqrt2
                    path_values[reality] = {k:v.clamp(min=0,max=1).cpu() for k,v in path_values[reality].items()}
                    del path_values[reality]['f(g(x))']
                    del path_values[reality]['f0(g(x))']
                    # del path_values[reality]['emboss(x)']

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
                plot_name += k+'|'
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

        super().__init__(*args, **kwargs)


    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):

        loss_types = ["main","lwf","sig"]
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, batch_mean=False)
        if realities[0].name == 'train_undist':
            loss_dict.pop("lwf")
            loss_dict.pop("sig")
            loss_dict["main"] = loss_dict["main"].mean() * 0.1
        elif realities[0].name == 'train_dist':
            loss_dict.pop("main")
            loss_dict["lwf"] = loss_dict["lwf"].mean() * 100.0
            loss_dict["sig"] = loss_dict["sig"].mean()

        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)


