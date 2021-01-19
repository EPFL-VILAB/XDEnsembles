
import numpy as np
import random, sys, os, time, glob, math, itertools, yaml, pickle
import parse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from PIL import ImageFilter
from torchvision import transforms

from functools import partial
from scipy import ndimage

import copy

import IPython

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

EXPERIMENT, BASE_DIR = open("config/jobinfo.txt").read().strip().split(', ')
JOB = "_".join(EXPERIMENT.split("_")[0:-1])

# MODELS_DIR = f"{BASE_DIR}/consistency_shared/models"
MODELS_DIR = f"/workspace/models"
DATA_DIRS = [f"/taskonomy-data/taskonomydata"]
RESULTS_DIR = f"/workspace/shared/results_{EXPERIMENT}"
SHARED_DIR = f"{BASE_DIR}/shared"
OOD_DIR = f"./assets/ood_natural/"
USE_RAID = False

os.system(f"mkdir -p {RESULTS_DIR}")


def both(x, y):
    x = dict(x.items())
    x.update(y)
    return x

def elapsed(last_time=[time.time()]):
    """ Returns the time passed since elapsed() was last called. """
    current_time = time.time()
    diff = current_time - last_time[0]
    last_time[0] = current_time
    return diff

def cycle(iterable):
    """ Cycles through iterable without making extra copies. """
    while True:
        for i in iterable:
            yield i

def average(arr):
    return sum(arr) / len(arr)

# def random_resize(iterable, vals=[128, 192, 256, 320]):
#    """ Cycles through iterable while randomly resizing batch values. """
#     from transforms import resize
#     while True:
#         for X, Y in iterable:
#             val = random.choice(vals)
#             yield resize(X.to(DEVICE), val=val).detach(), resize(Y.to(DEVICE), val=val).detach()


def get_files(exp, data_dirs=DATA_DIRS, recursive=False):
    """ Gets data files across mounted directories matching glob expression pattern. """
    # cache = SHARED_DIR + "/filecache_" + "_".join(exp.split()).replace(".", "_").replace("/", "_").replace("*", "_") + ("r" if recursive else "f") + ".pkl"
    # print ("Cache file: ", cache)
    # if os.path.exists(cache):
    #     return pickle.load(open(cache, 'rb'))

    files, seen = [], set()
    for data_dir in data_dirs:
        for file in glob.glob(f'{data_dir}/{exp}', recursive=recursive):
            if file[len(data_dir):] not in seen:
                files.append(file)
                seen.add(file[len(data_dir):])

    # pickle.dump(files, open(cache, 'wb'))
    return files


def get_finetuned_model_path(parents):
    if BASE_DIR == "/":
        return f"{RESULTS_DIR}/" + "_".join([parent.name for parent in parents[::-1]]) + ".pth"
    else:
        return f"{MODELS_DIR}/finetuned/" + "_".join([parent.name for parent in parents[::-1]]) + ".pth"


def plot_images(model, logger, test_set, dest_task="normal",
        ood_images=None, show_masks=False, loss_models={},
        preds_name=None, target_name=None, ood_name=None,
    ):

    from task_configs import get_task, ImageTask

    test_images, preds, targets, losses, _ = model.predict_with_data(test_set)

    if isinstance(dest_task, str):
        dest_task = get_task(dest_task)

    if show_masks and isinstance(dest_task, ImageTask):
        test_masks = ImageTask.build_mask(targets, dest_task.mask_val, tol=1e-3)
        logger.images(test_masks.float(), f"{dest_task}_masks", resize=64)

    dest_task.plot_func(preds, preds_name or f"{dest_task.name}_preds", logger)
    dest_task.plot_func(targets, target_name or f"{dest_task.name}_target", logger)

    if ood_images is not None:
        ood_preds = model.predict(ood_images)
        dest_task.plot_func(ood_preds, ood_name or f"{dest_task.name}_ood_preds", logger)

    for name, loss_model in loss_models.items():
        with torch.no_grad():
            output = loss_model(preds, targets, test_images)
            if hasattr(output, "task"):
                output.task.plot_func(output, name, logger, resize=128)
            else:
                logger.images(output.clamp(min=0, max=1), name, resize=128)


def gaussian_filter(channels=3, kernel_size=5, sigma=1.0, device=0):

    x_cord = torch.arange(kernel_size).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel


def motion_blur_filter(kernel_size=15):
    channels = 3
    kernel_motion_blur = torch.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = torch.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    kernel_motion_blur = kernel_motion_blur.view(1, 1, kernel_size, kernel_size)
    kernel_motion_blur = kernel_motion_blur.repeat(channels, 1, 1, 1)
    return kernel_motion_blur


def sobel_kernel(x):
    def sobel_transform(x):
        image = x.data.cpu().numpy().mean(axis=0)
        blur = ndimage.filters.gaussian_filter(image, sigma=2, )
        sx = ndimage.sobel(blur, axis=0, mode='constant')
        sy = ndimage.sobel(blur, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        edge = torch.FloatTensor(sob).unsqueeze(0)
        return edge

    x = torch.stack([sobel_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()


def binarized_kernel(x):
    def binarized_transform(x):
        image = (x>0.5)*1.0
        return image.float()
    x = torch.stack([binarized_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()  


class SobelKernel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return sobel_kernel(x)


def laplace_kernel(x):
    def laplace_transform(x):
        image = x.data.cpu().numpy().mean(axis=0) 
        blur = ndimage.filters.gaussian_filter(image, sigma=2, )
        lap = ndimage.laplace(blur) 
        edge = torch.FloatTensor(lap).unsqueeze(0)
        return edge
    x = torch.stack([laplace_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()

def gauss_kernel(x):
    def gauss_transform(x):
        x_cpu = x.data.cpu().numpy()
        r, g, b = x_cpu[0,:], x_cpu[1,:], x_cpu[2,:]
        fr, fg, fb = ndimage.filters.gaussian_filter(r, sigma=4), ndimage.filters.gaussian_filter(g, sigma=4), ndimage.filters.gaussian_filter(b, sigma=4)
        fr, fg, fb = fr[None,:], fg[None,:], fb[None,:]
        x_f = np.concatenate( (fr,fg,fb), axis=0)
        image = torch.FloatTensor(x_f)
        return image
    x = torch.stack([gauss_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()


emboss_weights = torch.tensor([[0.,0.,0.],[0.,1.0,0.],[-1.,0.,0.]])
emboss_weights = emboss_weights.view(1,1,3,3).cuda()
def emboss_kernel(x):
    def emboss_transform(x):
        x = x.mean(0,keepdim=True)
        x = (x*255).round().unsqueeze(0)
        image = F.conv2d(x,emboss_weights,padding=1)
        image = image + 128.0
        image = image.clamp(min=0.0,max=255.0)
        image = image / 255.0
        return image.squeeze(0)
    x = torch.stack([emboss_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()

# def emboss_kernel(x):
#     def emboss_transform(x):
#         x = x.mean(0,keepdim=True)
#         image = transforms.ToPILImage()(x.cpu())
#         imageEmboss = image.filter(ImageFilter.EMBOSS)
#         image = transforms.ToTensor()(imageEmboss)
    
#         return image

#     x = torch.stack([emboss_transform(y) for y in x], dim=0)
#     return x.to(DEVICE).requires_grad_() 

def greyscale(x):
    def grey_transform(x):
        return x.mean(0,keepdim=True)
    x = torch.stack([grey_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()


from pytorch_wavelets import DWTForward, DWTInverse

xfm = DWTForward(J=3, mode='zero', wave='db1').cuda()

def wav_kernel(x):
    def wav_transform(x):
        x_h, x_l = xfm(x.unsqueeze(0))
        x_h = F.interpolate(x_h, size=256, mode='bilinear')
        x_l_0, x_l_1, x_l_2 = F.interpolate(x_l[0][:,:,0,:], size=256, mode='bilinear'), F.interpolate(x_l[1][:,:,0,:], size=256, mode='bilinear') , F.interpolate(x_l[2][:,:,0,:], size=256, mode='bilinear')
        x_final = torch.cat((x_h.squeeze(),x_l_0.squeeze(),x_l_1.squeeze(),x_l_2.squeeze()), dim=0)

        return x_final

    x = torch.stack([wav_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()

emboss_weights = torch.tensor([[0.,0.,0.],[0.,1.0,0.],[-1.,0.,0.]])
emboss_weights = emboss_weights.view(1,1,3,3).cuda()
emboss_weights_2 = torch.tensor([[0.,0.,0.],[0.,1.0,0.],[0.,-1.,0.]])
emboss_weights_2 = emboss_weights_2.view(1,1,3,3).cuda()
emboss_weights_3 = torch.tensor([[0.,0.,0.],[0.,1.0,0.],[0.,0.,-1.]])
emboss_weights_3 = emboss_weights_3.view(1,1,3,3).cuda()
emboss_weights_4 = torch.tensor([[0.,0.,0.],[0.,1.0,-1.],[0.,0.,0.]])
emboss_weights_4 = emboss_weights_4.view(1,1,3,3).cuda()
def emboss4d_kernel(x):
    def emboss4d_transform(x):
        x = x.mean(0,keepdim=True)
        x = (x*255).round().unsqueeze(0)
        
        image1, image2, image3, image4 = F.conv2d(x,emboss_weights,padding=1),F.conv2d(x,emboss_weights_2,padding=1),F.conv2d(x,emboss_weights_3,padding=1),F.conv2d(x,emboss_weights_4,padding=1)
        image1, image2, image3, image4 = image1 + 128.0, image2 + 128.0, image3 + 128.0, image4 + 128.0
        image1, image2, image3, image4 = image1.clamp(min=0.0,max=255.0), image2.clamp(min=0.0,max=255.0), image3.clamp(min=0.0,max=255.0), image4.clamp(min=0.0,max=255.0)
        image1, image2, image3, image4 = image1 / 255.0, image2 / 255.0, image3 / 255.0, image4 / 255.0
    
        image = torch.cat((image1,image2,image3,image4), dim=1)
        return image.squeeze(0)
    x = torch.stack([emboss4d_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()


def get_gauss_kernel(sigma):
    
    truncate = 4.0

    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
        
    gauss_filt = np.array(weights)[:,None]@np.array(weights)[:,None].T
    gauss_weights = torch.from_numpy(gauss_filt).to(dtype=torch.float32)
    gauss_weights = gauss_weights.view(1,1,len(weights),-1).cuda()
    gauss_weights = gauss_weights/gauss_weights.sum()
    
    return gauss_weights

    
sigma_laplace_middomain = 2
gauss_kernel_laplace = get_gauss_kernel(sigma_laplace_middomain)
laplace_middomain_pad = nn.ReflectionPad2d(8)
laplace_middomain_pad2 = nn.ReplicationPad2d(1)
laplace_weights = torch.tensor([[0.,1.,0.],[1.,-4.0,1.],[0.,1.,0.]])
laplace_weights = laplace_weights.view(1,1,3,3).cuda()
def laplace_kernel(x):
    def laplace_transform(x):
        image = x.mean(0,keepdim=True)
        image = laplace_middomain_pad(image.unsqueeze(0))
        blur = F.conv2d(image,gauss_kernel_laplace,padding=0)
        
        
        #blur = ndimage.filters.gaussian_filter(image, sigma=2, )
        #lap = ndimage.laplace(blur) 
        #edge = torch.FloatTensor(lap).unsqueeze(0)
        
        blur = laplace_middomain_pad2(blur)
        
        edge = F.conv2d(blur,laplace_weights,padding=0).squeeze(0)

        return edge
    x = torch.stack([laplace_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()
   


sigma_gauss_middomain = 4
gauss_kernel_middomain = get_gauss_kernel(sigma_gauss_middomain)
gauss_middomain_pad = nn.ReflectionPad2d(16)
def gauss_kernel(x):
    def gauss_transform(x):
        img_t = gauss_middomain_pad(x.unsqueeze(0))
        r, g, b = img_t[:,0,:].unsqueeze(0), img_t[:,1,:].unsqueeze(0), img_t[:,2,:].unsqueeze(0)
        fr, fg, fb = F.conv2d(r,gauss_kernel_middomain,padding=0),F.conv2d(g,gauss_kernel_middomain,padding=0),F.conv2d(b,gauss_kernel_middomain,padding=0)

        image = torch.cat( (fr.squeeze(0),fg.squeeze(0),fb.squeeze(0)), axis=0)
        #x_cpu = x.data.cpu().numpy()
        #r, g, b = x_cpu[0,:], x_cpu[1,:], x_cpu[2,:]
        #fr, fg, fb = ndimage.filters.gaussian_filter(r, sigma=4), ndimage.filters.gaussian_filter(g, sigma=4), ndimage.filters.gaussian_filter(b, sigma=4)
        #fr, fg, fb = fr[None,:], fg[None,:], fb[None,:]
        #x_f = np.concatenate( (fr,fg,fb), axis=0)
        #image = torch.FloatTensor(x_f)
        return image
    x = torch.stack([gauss_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()

def sharp_kernel(x):
    def sharp_transform(x):
        #x_cpu = x.data.cpu().numpy()
        #r, g, b = x_cpu[0,:], x_cpu[1,:], x_cpu[2,:]
        #fr, fg, fb = np.fft.fft2(r), np.fft.fft2(g), np.fft.fft2(b)
        #fr, fg, fb = ndimage.fourier_gaussian(fr, sigma=4), ndimage.fourier_gaussian(fg, sigma=4), ndimage.fourier_gaussian(fb, sigma=4)
        #fr, fg, fb = ndimage.filters.gaussian_filter(r, sigma=4), ndimage.filters.gaussian_filter(g, sigma=4), ndimage.filters.gaussian_filter(b, sigma=4)
    
        #fr, fg, fb = fr[None,:], fg[None,:], fb[None,:]
        #x_f = np.concatenate( (fr,fg,fb), axis=0)
        
        
        img_t = gauss_middomain_pad(x.unsqueeze(0))
        r, g, b = img_t[:,0,:].unsqueeze(0), img_t[:,1,:].unsqueeze(0), img_t[:,2,:].unsqueeze(0)
        fr, fg, fb = F.conv2d(r,gauss_kernel_middomain,padding=0),F.conv2d(g,gauss_kernel_middomain,padding=0),F.conv2d(b,gauss_kernel_middomain,padding=0)

        x_f = torch.cat( (fr.squeeze(0),fg.squeeze(0),fb.squeeze(0)), axis=0)
        
        
        #x_f = (x_cpu - x_f) + x_cpu
        x_f = (x - x_f) + x
        image = x_f.clamp(min=0.0,max=1.0)
        #result = x_f
        #result_rf, result_gf, result_bf = result[0,:], result[1,:], result[2,:]
        #result_r, result_g, result_b = np.fft.ifft2(result_rf).real, np.fft.ifft2(result_gf).real, np.fft.ifft2(result_bf).real
        #result_r, result_g, result_b = result_r[None,:], result_g[None,:], result_b[None,:]
        #image = np.concatenate( (result_r,result_g,result_b), axis=0)
        #image = torch.FloatTensor(image)
    
        
        return image

    x = torch.stack([sharp_transform(y) for y in x], dim=0)
    return x.to(DEVICE).requires_grad_()

def create_t0_graph(folder_name,in_domain,out_domain):
    graph=torch.load(folder_name+'graph.pth')
    graph_t0={}
    graph_t0[f"('{in_domain}', 'normal_t0')"] = copy.deepcopy(graph[f"('{in_domain}', '{out_domain}')"])
    torch.save(graph_t0,folder_name+'graph_t0.pth')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # cpu  vars
    torch.cuda.manual_seed_all(seed) # gpu vars

def get_torch_size(x,size1):
    size = list(x)
    size[1] = size1
    return torch.Size(size)
