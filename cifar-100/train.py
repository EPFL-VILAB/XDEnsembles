
""" Adapted from https://github.com/weiaicunzai/pytorch-cifar100
"""

'''
  Name: train.py
  Desc: Executes training of the rgb -> CIFAR-100 classes or mid domain -> CIFAR-100 classes with the cross entropy loss.
    Here are some options that may be specified for any model. If they have a
    default value, it is given at the end of the description in parens.
        Data: Standard train and test split for CIFAR-100 data. They are loaded in utils.py with the relevant PyTorch loaders using the functions get_training_dataloader and get_test_dataloader. If a middle domain is employed, the input RGBs are first transformed using the appropriate functions in utils.py.
        Logging:
            'checkpoint': Folder where checkpoints are saved. This can be changed from conf/global_settings.py.
        Training:
            'b': The size of each batch. (256)
            'domain': Specifies training with directly from rgb input or one of the middle domains. 
            'net': Specifies model architecture to be used.
        Optimization:
            'lr': The initial learning rate to use for the training. (0.1)
  Usage:
    python train.py -domain emboss -net resnet18

'''

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import pickle

from torch.utils.data import DataLoader

from conf import settings
from utils import get_mean_std, get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

import pdb 

import wandb
wandb.init(project="xdomain-ensembles", entity="robust_team")

def train(epoch):

    start = time.time()
    net.train()
    
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        labels = labels.cuda()
        images = images.cuda()
        

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1


        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))


    train_scheduler.step(epoch)
    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

    return loss.item()

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        images = images.cuda()
        labels = labels.cuda()


        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
    ))

    return test_loss/ len(cifar100_test_loader.dataset), correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-domain', type=str, required=True, help='choose middle domain: direct, emboss, grey, sharp, lowpass')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-b', type=int, default=256, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    args = parser.parse_args()

    net = get_network(args)

    CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD = get_mean_std(args)

    wandb.config.update({"net":args.net,"batch_size":args.b,"lr":args.lr,"path":args.domain})

    #data preprocessing:
    cifar100_training_loader = get_training_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=16,
        batch_size=args.b,
        domain=args.domain,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=16,
        batch_size=args.b,
        domain=args.domain,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.1) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    weights_name = args.domain
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, weights_name)

    for epoch in range(1, settings.EPOCH):

        
        train_loss=train(epoch)
        test_loss,test_acc = eval_training(epoch)

        wandb.log({"train loss": train_loss, "test loss": test_loss, "test acc": test_acc})


        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path+'.pth')


    torch.save(net.state_dict(), checkpoint_path+'.pth')
