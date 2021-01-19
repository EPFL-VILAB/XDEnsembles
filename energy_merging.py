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

   "merge_reshading": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "sobel(x)": [tasks.rgb, tasks.sobel_edges],
            "f(sobel(x))": [tasks.rgb, tasks.sobel_edges, tasks.reshading],
            "emboss4d(x)": [tasks.rgb, tasks.emboss4d],
            "f(emboss4d(x))": [tasks.rgb, tasks.emboss4d, tasks.reshading],
            "grey(x)": [tasks.rgb, tasks.grey],
            "f(grey(x))": [tasks.rgb, tasks.grey, tasks.reshading],
            "wav(x)": [tasks.rgb, tasks.wav],
            "f(wav(x))": [tasks.rgb, tasks.wav, tasks.reshading],
            "laplace(x)": [tasks.rgb, tasks.laplace_edges],
            "f(laplace(x))": [tasks.rgb, tasks.laplace_edges, tasks.reshading],
            "sharp(x)": [tasks.rgb, tasks.sharp],
            "f(sharp(x))": [tasks.rgb, tasks.sharp, tasks.reshading],
            "gauss(x)": [tasks.rgb, tasks.gauss],
            "f(gauss(x))": [tasks.rgb, tasks.gauss, tasks.reshading],
            "n(x)": [tasks.rgb, tasks.reshading],
            "stacked2reshade(x)": [tasks.stackedr, tasks.reshading],
        },
        "freeze_list": [[tasks.sobel_edges, tasks.reshading],
                        [tasks.grey, tasks.reshading],
                        [tasks.emboss4d, tasks.reshading],
                        [tasks.wav, tasks.reshading],
                        [tasks.rgb, tasks.reshading],
                        [tasks.laplace_edges, tasks.reshading],
                        [tasks.sharp, tasks.reshading],
                        [tasks.gauss, tasks.reshading]],
        "losses": {
            "main": {
                ("train_c", "val_c", "val"): [
                    ("stacked2reshade(x)", "y^"),
                ],
            },
            "path1": {
                ("train_c", "val_c", "val"): [
                    ("f(sobel(x))", "y^"),
                ],
            },
            "path2": {
                ("train_c", "val_c", "val"): [
                    ("f(emboss4d(x))", "y^"),
                ],
            },
            "path3": {
                ("train_c", "val_c", "val"): [
                    ("f(grey(x))", "y^"),
                ],
            },
            "path4": {
                ("train_c", "val_c", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "path5": {
                ("train_c", "val_c", "val"): [
                    ("f(wav(x))", "y^"),
                ],
            },
            "path6": {
                ("train_c", "val_c", "val"): [
                    ("f(laplace(x))", "y^"),
                ],
            },
            "path7": {
                ("train_c", "val_c", "val"): [
                    ("f(sharp(x))", "y^"),
                ],
            },
            "path8": {
                ("train_c", "val_c", "val"): [
                    ("f(gauss(x))", "y^"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood", "ood_syn_aug"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "sobel(x)",
                    "f(sobel(x))",
                    "emboss4d(x)",
                    "f(emboss4d(x))",
                    "grey(x)",
                    "f(grey(x))",
                    "wav(x)",
                    "f(wav(x))",
                    "laplace(x)",
                    "f(laplace(x))",
                    "sharp(x)",
                    "f(sharp(x))",
                    "gauss(x)",
                    "f(gauss(x))",
                    "stacked2reshade(x)"
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
        paths = dict(sorted(paths.items()))
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

        all_tasks = list(set(tasks))
        valid_tasks = [x for x in all_tasks if x.name!='stackedr']
        return valid_tasks #list(set(tasks))

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, batch_mean=True, use_l1=False):
        
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

            paths_ = ['f(emboss4d(x))','f(grey(x))','f(sobel(x))','f(wav(x))','n(x)', 'f(gauss(x))', 'f(laplace(x))', 'f(sharp(x))']
            npaths = len(paths_)
            pred_size = get_torch_size(path_values['n(x)'].size(),2*npaths+npaths)
            # to get mixture pdf evaled at each components mu
            mus = torch.cat((path_values[paths_[0]][:,:1],path_values[paths_[1]][:,:1],path_values[paths_[2]][:,:1],path_values[paths_[3]][:,:1],path_values[paths_[4]][:,:1],path_values[paths_[5]][:,:1],path_values[paths_[6]][:,:1],path_values[paths_[7]][:,:1]),dim=1)
            sigs = torch.cat((path_values[paths_[0]][:,1:],path_values[paths_[1]][:,1:],path_values[paths_[2]][:,1:],path_values[paths_[3]][:,1:],path_values[paths_[4]][:,1:],path_values[paths_[5]][:,1:],path_values[paths_[6]][:,1:],path_values[paths_[7]][:,1:]),dim=1).exp()
            lap_dist = torch.distributions.Laplace(loc=mus, scale=sigs+1e-15)
            pdfs = []
            for i in range(npaths):
                pdfs.append((lap_dist.log_prob(path_values[paths_[i]][:,:1]).exp() * path_values['stacked2reshade(x)']).sum(1,keepdim=True))
            pi = torch.cat(pdfs,dim=1)
            

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
                        nchannels = 1
                        ## nll loss
                        if loss_type=='main':    # for stacked output nll loss
                            pred = torch.cuda.FloatTensor(pred_size)
                            pred[:,:npaths] = mus
                            pred[:,npaths:npaths*2] = sigs.log()
                            pred[:,-npaths:] = path_values['stacked2reshade(x)']    # weights
                            path_loss_nll, _ = output_task.nll(pred, path_values[path2][:,:1], batch_mean=batch_mean, compute_mask=True)
                            loss[loss_type] += path_loss_nll
                            self.metrics[reality.name][loss_type + "_nll : "+path1 + " -> " + path2] += [path_loss_nll.mean().detach().cpu()]
                        elif 'path' in loss_type:   # for individual paths nll loss
                            pred = path_values[path1]
                            path_loss_nll, _ = output_task.nll(pred, path_values[path2][:,:1], batch_mean=batch_mean, compute_mask=True)
                            self.metrics[reality.name][loss_type + "_nll : "+path1 + " -> " + path2] += [path_loss_nll.mean().detach().cpu()]
                        ## l norm loss
                        if loss_type=='main':
                            onehot = torch.cuda.FloatTensor(pi.size()).fill_(0.)
                            onehot.scatter_(1,pi.argmax(1,keepdim=True),1.0)
                            pred = (pred[:,:npaths] * onehot).sum(1,keepdim=True)
                        elif 'path' in loss_type:
                            pred = path_values[path1][:,:nchannels]
                        elif loss_type=='mode':
                            pi = pi / pi.sum(1,keepdim=True)
                            pred = (F.gumbel_softmax((pi+1e-10).log(),tau=0.2,dim=1) * mus).sum(dim=1,keepdim=True)
                        path_loss, _ = output_task.norm(pred, path_values[path2][:,:nchannels], batch_mean=batch_mean, compute_mask=True, compute_mse=False)
                        if loss_type=='mode': loss[loss_type] += path_loss
                        self.metrics[reality.name][loss_type + "_mae : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]
                        path_loss, _ = output_task.norm(pred, path_values[path2][:,:nchannels], batch_mean=batch_mean, compute_mask=True, compute_mse=True)
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
                    path_values[reality] = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])

                    pred_mu = torch.Tensor().cuda()
                    pred_sig = torch.Tensor().cuda()
                    paths_ = ['f(emboss4d(x))','f(grey(x))','f(sobel(x))','f(wav(x))','n(x)', 'f(gauss(x))', 'f(laplace(x))', 'f(sharp(x))']
                    for p in paths_:
                        if p in ['x','y^','bin(x)','grey(x)','emboss(x)','sobel(x)','stacked2reshade(x)', 'gauss(x)', 'laplace(x)', 'sharp(x)']: continue
                        if 'y^' in p and reality == 'ood': continue
                        path_values[reality][f'{p}_m'] = path_values[reality][p][:,:1]
                        pred_mu = torch.cat((pred_mu,path_values[reality][f'{p}_m']),dim=1)
                        path_values[reality][f'{p}_s'] = path_values[reality][p][:,1:2].exp()*sqrt2
                        pred_sig = torch.cat((pred_sig,path_values[reality][f'{p}_s']),dim=1)
                        del path_values[reality][p]

                    lap_dist = torch.distributions.Laplace(loc=pred_mu, scale=pred_sig+1e-15)
                    pdfs = []
                    for i in range(len(paths_)):
                        pdfs.append((lap_dist.log_prob(path_values[reality][f'{paths_[i]}_m']).exp() * path_values[reality]['stacked2reshade(x)']).sum(1,keepdim=True))
                    pi = torch.cat(pdfs,dim=1)
                    onehot = torch.cuda.FloatTensor(pi.size()).fill_(0.)
                    onehot.scatter_(1,pi.argmax(1,keepdim=True),1.0)
                    path_values[reality][f'stacked2reshade(x)_m'] = (pred_mu * onehot).sum(1,keepdim=True)
                    path_values[reality][f'stacked2reshade(x)_s'] = (pred_sig * onehot).sum(1,keepdim=True)

                    for i in [4,0,1,2,3, 5, 6, 7]:
                        path_values[reality][f'stacked2reshade(x)_w{i+1}'] = path_values[reality][f'stacked2reshade(x)'][:,i:i+1]                     
                    del path_values[reality]['stacked2reshade(x)']

                    path_values[reality]['emboss4d(x)'] = path_values[reality]['emboss4d(x)'][:,:3]

                    if reality is 'test': #compute error map
                        mask_task = self.paths["y^"][-1]
                        mask = ImageTask.build_mask(path_values[reality]["y^"], val=mask_task.mask_val)
                        errors = ((path_values[reality]["y^"][:,:1]-path_values[reality]["stacked2reshade(x)_m"][:,:1])**2).mean(dim=1, keepdim=True)
                        errors = (3*errors/(mask_task.variance)).clamp(min=0, max=1)
                        log_errors = torch.log(errors + 1)
                        log_errors = log_errors / log_errors.max()
                        log_errors = torch.tensor(cmap(log_errors.cpu()))[:, 0].permute((0, 3, 1, 2)).float()[:, 0:3]
                        log_errors = log_errors.clamp(min=0, max=1).to(DEVICE)
                        log_errors[~mask.expand_as(log_errors)] = 0.505
                        path_values[reality]['error']= log_errors

                    path_values[reality] = {k:v.clamp(min=0,max=1).cpu() for k,v in path_values[reality].items()}

                    del path_values[reality][f'wav(x)']

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
            
            keys_list = list(path_values[reality].keys())
            keys_list.remove('x')
            keys_list.insert(0,'x')
            if 'y^' in keys_list: 
                keys_list.remove('y^')
                keys_list.insert(1,'y^')
            if 'error' in keys_list:
                keys_list.remove('error')
                keys_list.insert(-5,'error')

            for k in keys_list:
                plot_name += k+'|'
                img_row = reshape_img_to_rows(path_values[reality][k])
                if img_row.size(0) == 1: img_row = img_row.repeat(3,1,1)
                all_imgs_reality.append(img_row)
            plot_name = plot_name[:-1]
            plot_name = plot_name.replace("stacked2reshade","st")
            plot_name = plot_name.replace("emboss4d","em")
            plot_name = plot_name.replace("grey","gr")
            plot_name = plot_name.replace("sobel","sb")
            plot_name = plot_name.replace("laplace","lp")
            plot_name = plot_name.replace("gauss","gs")
            plot_name = plot_name.replace("(x)","")
            all_images[reality] = torch.cat(all_imgs_reality,dim=-2)

        return all_images


    def __repr__(self):
        return str(self.losses)


class EnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.random_select = kwargs.pop('random_select', False)
        self.running_stats = {}
        self.target_task = kwargs['paths']['y^'][0].name

        super().__init__(*args, **kwargs)


    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, compute_grad_ratio=False):

        loss_types = ["main","path1","path2","path3","path4","path5","path6","path7","path8"]
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, batch_mean=False)
        loss_dict["main"] = loss_dict["main"].mean() * 0.1
        for i in range(1,9):
            loss_dict.pop("path"+str(i))

        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)


