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


SQRT2 = math.sqrt(2)

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

    ### direct 

    "baseline_rgb2normal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.normal],
            "normal": [tasks.normal],
        },
        "freeze_list": [],
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("f(n(x))", "y^"),
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
                    "f(n(x))",
                ]
            ),
        },
    },

    "baseline_rgb2reshade": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "n(x)": [tasks.rgb, tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.reshading],
            "reshade": [tasks.reshading],
        },
        "freeze_list": [],
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
                    "f(n(x))",
                ]
            ),
        },
    },

    "baseline_rgb2depth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.depth_zbuffer],
            "depth": [tasks.depth_zbuffer],
        },
        "freeze_list": [],
        "losses": {
            "nll": {
                ("train", "val"): [
                    ("f(n(x))", "y^"),
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
                    "f(n(x))",
                ]
            ),
        },
    },

    ### mid-domain to depth 

    "baseline_edge2depth": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.depth_zbuffer],
        },
        "freeze_list": [[tasks.rgb, tasks.sobel_edges]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","depth"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "n(x)",
                    "f(n(x))",
                    "depth"
                ]
            ),
        },
    },

    "baseline_grey2depth": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.grey],
            "depth": [tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.grey, tasks.depth_zbuffer],
        },
        "freeze_list": [[tasks.rgb, tasks.grey]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","depth"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "n(x)",
                    "f(n(x))",
                    "depth"
                ]
            ),
        },
    },

    "baseline_wav2depth": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.wav],
            "depth": [tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.wav, tasks.depth_zbuffer],
        },
        "freeze_list": [[tasks.rgb, tasks.wav]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","depth"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    # "n(x)",
                    "f(n(x))",
                    "depth"
                ]
            ),
        },
    },

    "baseline_emboss2depth": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.emboss4d],
            "depth": [tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.emboss4d, tasks.depth_zbuffer],
        },
        "freeze_list": [[tasks.rgb, tasks.emboss4d]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","depth"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "n(x)",
                    "f(n(x))",
                    "depth"
                ]
            ),
        },
    },

    "baseline_gauss2depth": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.gauss],
            "depth": [tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.gauss, tasks.depth_zbuffer],
        },
        "freeze_list": [[tasks.rgb, tasks.gauss]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","depth"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "n(x)",
                    "f(n(x))",
                    "depth"
                ]
            ),
        },
    },

    "baseline_lap2depth": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.laplace_edges],
            "depth": [tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.depth_zbuffer],
        },
        "freeze_list": [[tasks.rgb, tasks.laplace_edges]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","depth"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "n(x)",
                    "f(n(x))",
                    "depth"
                ]
            ),
        },
    },

    "baseline_sharp2depth": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.sharp],
            "depth": [tasks.depth_zbuffer],
            "f(n(x))": [tasks.rgb, tasks.sharp, tasks.depth_zbuffer],
        },
        "freeze_list": [[tasks.rgb, tasks.sharp]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","depth"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "n(x)",
                    "f(n(x))",
                    "depth"
                ]
            ),
        },
    },

    ### mid-domain to reshade

    "baseline_emboss2reshade": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.emboss4d],
            "reshade": [tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.emboss4d, tasks.reshading],
        },
        "freeze_list": [[tasks.rgb, tasks.emboss4d]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","reshade"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "n(x)",
                    "f(n(x))",
                    "reshade"
                ]
            ),
        },
    },

    "baseline_gauss2reshade": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.gauss],
            "reshade": [tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.gauss, tasks.reshading],
        },
        "freeze_list": [[tasks.rgb, tasks.gauss]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","reshade"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "reshade"
                ]
            ),
        },
    },

    "baseline_edge2reshade": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.sobel_edges],
            "reshade": [tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.reshading],
        },
        "freeze_list": [[tasks.rgb, tasks.sobel_edges]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","reshade"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "reshade"
                ]
            ),
        },
    },

    "baseline_lap2reshade": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.laplace_edges],
            "reshade": [tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.reshading],
        },
        "freeze_list": [[tasks.rgb, tasks.laplace_edges]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","reshade"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "reshade"
                ]
            ),
        },
    },

    "baseline_sharp2reshade": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.sharp],
            "reshade": [tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.sharp, tasks.reshading],
        },
        "freeze_list": [[tasks.rgb, tasks.sharp]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","reshade"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "reshade"
                ]
            ),
        },
    },

    "baseline_gauss2reshade": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.gauss],
            "reshade": [tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.gauss, tasks.reshading],
        },
        "freeze_list": [[tasks.rgb, tasks.gauss]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","reshade"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "reshade"
                ]
            ),
        },
    },

    "baseline_wav2reshade": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.wav],
            "reshade": [tasks.reshading],
            "f(n(x))": [tasks.rgb, tasks.wav, tasks.reshading],
        },
        "freeze_list": [[tasks.rgb, tasks.wav]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","reshade"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    # "n(x)",
                    "f(n(x))",
                    "reshade"
                ]
            ),
        },
    },

    ### mid-domain to normal
    "baseline_emboss2normal": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.emboss4d],
            "normal": [tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.emboss4d, tasks.normal],
        },
        "freeze_list": [[tasks.rgb, tasks.emboss4d]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","normal"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "n(x)",
                    "f(n(x))",
                    "normal"
                ]
            ),
        },
    },

    "baseline_gauss2normal": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.gauss],
            "normal": [tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.gauss, tasks.normal],
        },
        "freeze_list": [[tasks.rgb, tasks.gauss]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","normal"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "normal"
                ]
            ),
        },
    },

    "baseline_edge2normal": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.sobel_edges],
            "normal": [tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal],
        },
        "freeze_list": [[tasks.rgb, tasks.sobel_edges]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","normal"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "normal"
                ]
            ),
        },
    },

    "baseline_lap2normal": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.laplace_edges],
            "normal": [tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.laplace_edges, tasks.normal],
        },
        "freeze_list": [[tasks.rgb, tasks.laplace_edges]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","normal"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "normal"
                ]
            ),
        },
    },

    "baseline_sharp2normal": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.sharp],
            "normal": [tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.sharp, tasks.normal],
        },
        "freeze_list": [[tasks.rgb, tasks.sharp]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","normal"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "normal"
                ]
            ),
        },
    },

    "baseline_gauss2normal": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.gauss],
            "normal": [tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.gauss, tasks.normal],
        },
        "freeze_list": [[tasks.rgb, tasks.gauss]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","normal"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "n(x)",
                    "f(n(x))",
                    "normal"
                ]
            ),
        },
    },

    "baseline_wav2normal": {
        "paths": {
            "x": [tasks.rgb],
            "n(x)": [tasks.rgb,tasks.wav],
            "normal": [tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.wav, tasks.normal],
        },
        "freeze_list": [[tasks.rgb, tasks.wav]],
        "losses": {
            "nll": {
                ("train","val"): [
                    ("f(n(x))","normal"),
                ]
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    # "n(x)",
                    "f(n(x))",
                    "normal"
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
        # pdb.set_trace()
        paths = dict(sorted(paths.items(), key = lambda item : len(item[1])))
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
            # pdb.set_trace()
            path_values = self.compute_paths(graph,
                paths={
                    path: self.paths[path] for path in \
                    set(path for paths in losses for path in paths)
                    },
                reality=reality)

            if reality.name not in self.metrics:
                self.metrics[reality.name] = defaultdict(list)

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
                        path_loss_nll, _ = output_task.nll(path_values[path1], path_values[path2], batch_mean=batch_mean, compute_mask=True)
                        loss[loss_type] += path_loss_nll
                        self.metrics[reality.name]["nll : "+path1 + " -> " + path2] += [path_loss_nll.mean().detach().cpu()]
                        path_loss, _ = output_task.norm(path_values[path1][:,:1], path_values[path2], batch_mean=batch_mean, compute_mask=True, compute_mse=False)
                        # pdb.set_trace()
                        self.metrics[reality.name]["mae : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]
                        path_loss, _ = output_task.norm(path_values[path1][:,:1], path_values[path2], batch_mean=batch_mean, compute_mask=True, compute_mse=True)
                        self.metrics[reality.name]["mse : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]
        # pdb.set_trace()
        # del loss["nll2"]
        # del loss["nll3"]

        return loss

    def logger_hooks(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
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

                    nc = path_values[reality]['f(n(x))'].size(1)//2
                    path_values[reality][f'f(n(x))_m'] = path_values[reality]['f(n(x))'][:,:nc]
                    path_values[reality][f'f(n(x))_s'] = path_values[reality]['f(n(x))'][:,nc:].exp()*SQRT2
                    
                    path_values[reality] = {k:v.clamp(min=0,max=1).cpu() for k,v in path_values[reality].items()}
                    path_values[reality].pop('f(n(x))', None)

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
            all_images[reality+'_'+plot_name] = torch.cat(all_imgs_reality,dim=-2)

        return all_images


    def __repr__(self):
        return str(self.losses)


class EnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 3)
        # self.random_select = kwargs.pop('random_select', False)
        self.running_stats = {}
        self.target_task = kwargs['paths']['n(x)'][0].name

        super().__init__(*args, **kwargs)


    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, compute_grad_ratio=False):

        loss_types = ["nll"]
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, batch_mean=False)
        loss_dict["nll"] = loss_dict["nll"].mean()
        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)


