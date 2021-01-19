
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from scipy import ndimage
import torch.nn as nn
import math
import torch.nn.functional as F

from functools import partial

import pdb


def get_mean_std(args):
    if args.domain == 'direct':
        #mean and std of cifar100 dataset direct
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    if args.domain == 'emboss':
        #mean and std of cifar100 dataset emboss
        CIFAR100_TRAIN_MEAN = (0.608645, 0.5112544, 0.51275474, 0.51249)
        CIFAR100_TRAIN_STD = (0.19839302, 0.112616025, 0.113838084, 0.115355805)

    if args.domain == 'lowpass':
        #mean and std of cifar100 dataset lowpass
        CIFAR100_TRAIN_MEAN = (0.49573156, 0.49705598, 0.49573156)
        CIFAR100_TRAIN_STD = (0.2759571, 0.28106198, 0.2759571)

    if args.domain == 'grey':
        #mean and std of cifar100 grey
        CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        #CIFAR100_TRAIN_MEAN = (0.4781806253063725)
        CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        #CIFAR100_TRAIN_STD = (0.26664107337400417)

    if args.domain == 'sharp':
        #mean and std of cifar100 dataset sharp
        CIFAR100_TRAIN_MEAN = (0.5050364, 0.4905038, 0.48903266)
        CIFAR100_TRAIN_STD = (0.3087077, 0.2999974, 0.2968065)

    return CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD

def get_network(args):
    """ return given network
    """

    if args.domain=='emboss':
        in_ch = 4
    elif args.domain=='grey':
        in_ch = 1
    else:
        in_ch = 3

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(in_ch=in_ch)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    # if args.gpu: #use_gpu
    net = net.cuda()

    return net

def get_gauss_kernel(kernel_size = 9,sigma = 2.):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2.*variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    return gaussian_kernel

rpad=nn.ReflectionPad2d(8)
sobel_weights_x = torch.tensor([[[[1., 2., 1.], [0., 0., 0.], [-1., -2., -1]]]])
sobel_weights_y = sobel_weights_x.permute(0,1,3,2)
gauss_weights = get_gauss_kernel(kernel_size = 2)
def sobel_kernel(x):
    def sobel_transform(x):
        x = x.mean(0,keepdim=True)
        x = x.unsqueeze(0)
        x = rpad(x)
        image = F.conv2d(x,gauss_weights)
        image_x = F.conv2d(image,sobel_weights_x,padding=1)
        image_y = F.conv2d(image,sobel_weights_y,padding=1)
        image = (image_x**2 + image_y**2+1e-10).sqrt()
        return image.squeeze(0)
#     x = torch.stack([sobel_transform(y) for y in x], dim=0)
    x = sobel_transform(x)
    x = F.interpolate(x.unsqueeze(0), size=(32,32))[0]
    return x

sigma_gauss_middomain = 3
gauss_kernel_middomain = get_gauss_kernel(sigma_gauss_middomain)
gauss_middomain_pad_sharp = nn.ReflectionPad2d(1)
def sharp_kernel(x):
    def sharp_transform(x):
        
        img_t = gauss_middomain_pad_sharp(x.unsqueeze(0))
        r, g, b = img_t[:,0,:].unsqueeze(0), img_t[:,1,:].unsqueeze(0), img_t[:,2,:].unsqueeze(0)
        fr, fg, fb = F.conv2d(r,gauss_kernel_middomain,padding=0),F.conv2d(g,gauss_kernel_middomain,padding=0),F.conv2d(b,gauss_kernel_middomain,padding=0)

        x_f = torch.cat( (fr.squeeze(0),fg.squeeze(0),fb.squeeze(0)), axis=0)

        x_f = (x - x_f) + x
        image = x_f.clamp(min=0.0,max=1.0)
        return image

    x = sharp_transform(x)
    return x

def greyscale(x):
    def grey_transform(x):
        return x.mean(0,keepdim=True)
    x = grey_transform(x)
    return x


emboss_weights = torch.tensor([[0.,0.,0.],[0.,1.0,0.],[-1.,0.,0.]])
emboss_weights = emboss_weights.view(1,1,3,3)
emboss_weights_2 = torch.tensor([[0.,0.,0.],[0.,1.0,0.],[0.,-1.,0.]])
emboss_weights_2 = emboss_weights_2.view(1,1,3,3)
emboss_weights_3 = torch.tensor([[0.,0.,0.],[0.,1.0,0.],[0.,0.,-1.]])
emboss_weights_3 = emboss_weights_3.view(1,1,3,3)
emboss_weights_4 = torch.tensor([[0.,0.,0.],[0.,1.0,-1.],[0.,0.,0.]])
emboss_weights_4 = emboss_weights_4.view(1,1,3,3)
def emboss4d_kernel(x):
    def emboss4d_transform(x):
        x = x.mean(0,keepdim=True)
        x = (x*255).unsqueeze(0)
        image1, image2, image3, image4 = F.conv2d(x,emboss_weights,padding=1),F.conv2d(x,emboss_weights_2,padding=1),F.conv2d(x,emboss_weights_3,padding=1),F.conv2d(x,emboss_weights_4,padding=1)
        image1, image2, image3, image4 = image1 + 128.0, image2 + 128.0, image3 + 128.0, image4 + 128.0
        image1, image2, image3, image4 = image1.clamp(min=0.0,max=255.0), image2.clamp(min=0.0,max=255.0), image3.clamp(min=0.0,max=255.0), image4.clamp(min=0.0,max=255.0)
        image1, image2, image3, image4 = image1 / 255.0, image2 / 255.0, image3 / 255.0, image4 / 255.0
    
        image = torch.cat((image1,image2,image3,image4), dim=1)
        return image.squeeze(0)
#     x = torch.stack([emboss4d_transform(y) for y in x], dim=0)
    x = emboss4d_transform(x)
    return x

sigma_gauss_middomain = 3
gauss_kernel_middomain = get_gauss_kernel(sigma_gauss_middomain)
gauss_middomain_pad = nn.ReflectionPad2d(0)
def gauss_kernel(x):
    def gauss_transform(x):
        img_t = gauss_middomain_pad(x.unsqueeze(0))
        r, g, b = img_t[:,0,:].unsqueeze(0), img_t[:,1,:].unsqueeze(0), img_t[:,2,:].unsqueeze(0)
        fr, fg, fb = F.conv2d(r,gauss_kernel_middomain,padding=1),F.conv2d(g,gauss_kernel_middomain,padding=1),F.conv2d(b,gauss_kernel_middomain,padding=1)
        image = torch.cat( (fr.squeeze(0),fg.squeeze(0),fb.squeeze(0)), axis=0)
        return image
#     x = torch.stack([gauss_transform(y) for y in x], dim=0)
    x = gauss_transform(x)
    # x = F.interpolate(x.unsqueeze(0), size=(32,32))[0]
    return x

def get_training_dataloader(mean, std, batch_size=16, num_workers=2, domain='direct', shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    if domain == 'emboss':
        mid_domain_transformation = partial(emboss4d_kernel)
    elif domain == 'sharp':
        mid_domain_transformation = partial(sharp_kernel)
    elif domain == 'lowpass':
        mid_domain_transformation = partial(gauss_kernel)
    elif domain == 'grey':
        mid_domain_transformation = partial(greyscale)

    if domain == 'direct':
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif domain == 'grey':
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            mid_domain_transformation
        ])
    else:
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            mid_domain_transformation,
            transforms.Normalize(mean, std)
        ])

    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    # mean, std = compute_mean_std(cifar100_training)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, domain='direct', shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    if domain == 'emboss':
        mid_domain_transformation = partial(emboss4d_kernel)
    elif domain == 'sharp':
        mid_domain_transformation = partial(sharp_kernel)
    elif domain == 'lowpass':
        mid_domain_transformation = partial(gauss_kernel)
    elif domain == 'grey':
        mid_domain_transformation = partial(greyscale)

    if domain == 'direct':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif domain == 'grey':
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        mid_domain_transformation
        ])
    else:
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        mid_domain_transformation,
        transforms.Normalize(mean, std)
        ])

    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][0][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][0][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][0][:, :, 2] for i in range(len(cifar100_dataset))])
    data_x = numpy.dstack([cifar100_dataset[i][0][:, :, 3] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b), numpy.mean(data_x)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b), numpy.std(data_x)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]