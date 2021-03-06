import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np
import IPython

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, WrapperModel
from datasets import TaskDataset
from task_configs import get_task, task_map, tasks, get_model, RealityTask
from transfers import Transfer, RealityTransfer, get_transfer_name


class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, tasks=tasks, edges=None, edges_exclude=None,
        pretrained=True, finetuned=False,
        reality=[], task_filter=[tasks.segment_semantic],
        freeze_list=[], lazy=False, initialize_from_transfer=True,
    ):

        super().__init__()
        self.tasks = list(set(tasks) - set(task_filter))
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.edge_list, self.edge_list_exclude = edges, edges_exclude
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges, self.adj, self.in_adj = [], defaultdict(list), defaultdict(list)
        self.edge_map, self.reality = {}, reality
        self.initialize_from_transfer = initialize_from_transfer
        print('graph tasks', self.tasks)
        self.params = {}

        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = (src_task, dest_task)
            if edges is not None and key not in edges: continue
            if edges_exclude is not None and key in edges_exclude: continue
            if src_task == dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            print (src_task, dest_task)
            transfer = None
            if isinstance(src_task, RealityTask):
                if dest_task not in src_task.tasks: continue
                transfer = RealityTransfer(src_task, dest_task)
            else:
                transfer = Transfer(src_task, dest_task,
                    pretrained=pretrained, finetuned=finetuned
                )
                transfer.name = get_transfer_name(transfer)
                if not self.initialize_from_transfer:
                    transfer.path = None
            if transfer.model_type is None:
                continue
            print ("Added transfer", transfer)
            self.edges += [transfer]
            self.adj[src_task.name] += [transfer]
            self.in_adj[dest_task.name] += [transfer]
            self.edge_map[str((src_task.name, dest_task.name))] = transfer
            if isinstance(transfer, nn.Module):
                if str((src_task.name, dest_task.name)) not in freeze_list:
                    self.params[str((src_task.name, dest_task.name))] = transfer
                else:
                    print("freezing " + str((src_task.name, dest_task.name)))
                try:
                    if not lazy: transfer.load_model()
                except:
                    IPython.embed()

        self.params = nn.ModuleDict(self.params)

    def edge(self, src_task, dest_task):
        key1 = str((src_task.name, dest_task.name))
        key2 = str((src_task.kind, dest_task.kind))
        if key1 in self.edge_map: return self.edge_map[key1]
        return self.edge_map[key2]

    def sample_path(self, path, reality=None, use_cache=False, cache={}, name=None):
        path = [reality or self.reality[0]] + path
        x = None
        for i in range(1, len(path)):
            try:

                ## only used in consistency training phase, 
                ## for passing correct channels to percep models
                if path[i-1].name == 'reshading':
                    if x.size(1)==2: 
                        x = x[:,:1].repeat(1,3,1,1)
                elif path[i-1].name == 'normal':
                    if x.size(1)==6: x = x[:,:3]
                elif path[i-1].name == 'depth_zbuffer':
                    if x.size(1)==2: x = x[:,:1]

                ## only used for training network merging
                if path[i].name == 'stackedr': 
                    x = torch.Tensor().cuda()
                    for k,v in cache.items():
                        if len(k)>=3 and k[-1].name in ['reshading','normal','depth_zbuffer']:
                            x = torch.cat((x,v),dim=1)
                    continue
                        
                x = cache.get(tuple(path[0:(i+1)]),
                    self.edge(path[i-1], path[i])(x)
                )
            except KeyError:
                return None
            except Exception as e:
                IPython.embed()

            if use_cache: cache[tuple(path[0:(i+1)])] = x
        return x

    def save(self, weights_file=None, weights_dir=None):

        ### TODO: save optimizers here too
        if weights_file:
            torch.save({
                key: model.state_dict() for key, model in self.edge_map.items() \
                if not isinstance(model, RealityTransfer)
            }, weights_file)

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            for key, model in self.edge_map.items():
                if isinstance(model, RealityTransfer): continue
                if not isinstance(model.model, TrainableModel): continue
                model.model.save(f"{weights_dir}/{model.name}.pth")
            torch.save(self.optimizer, f"{weights_dir}/optimizer.pth")

    def load_weights(self, weights_file=None, key_filter=None):
        for key, state_dict in torch.load(weights_file).items():
            if key in self.edge_map and (key_filter is None or key in key_filter):
                print('loading', key)
                self.edge_map[key].load_state_dict(state_dict)

#    def load_weights(self, weights_file=None):
#        for key, state_dict in torch.load(weights_file).items():
#            if key in self.edge_map:
#                self.edge_map[key].load_state_dict(state_dict)

    # def load_weights(self, weights_file=None):
    #     loaded_something = False
    #     for key, state_dict in torch.load(weights_file).items():
    #         if key in self.edge_map:
    #             loaded_something = True
    #             self.edge_map[key].load_model()
    #             self.edge_map[key].load_state_dict(state_dict)
    #     if not loaded_something:
    #         raise RuntimeError(f"No edges loaded from file: {weights_file}")
