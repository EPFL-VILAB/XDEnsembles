
import numpy as np
import random, sys, os, time, glob, math, itertools, json, copy
from collections import defaultdict, namedtuple
from functools import partial

import PIL
from PIL import Image
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torchvision import transforms

from utils import *
from models import DataParallelModel
from modules.unet import UNet, UNetOld2, UNetOld
from modules.percep_nets import Dense1by1Net
from modules.depth_nets import UNetDepth
from modules.resnet import ResNetClass
import IPython


from PIL import ImageFilter
from skimage.filters import gaussian
from distortions import *

import pdb

try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO

class GaussianBulr(object):
    def __init__(self, radius):
        self.radius = radius
        self.filter = ImageFilter.GaussianBlur(radius)

    def __call__(self, im):
        return im.filter(self.filter)

    def __repr__(self):
        return 'GaussianBulr Filter with Radius {:d}'.format(self.radius)


""" Model definitions for launching new transfer jobs between tasks. """
model_types = {
    ('normal', 'principal_curvature'): lambda : Dense1by1Net(),
    ('normal', 'depth_zbuffer'): lambda : UNetDepth(),
    ('normal', 'reshading'): lambda : UNet(downsample=5),
    ('depth_zbuffer', 'normal'): lambda : UNet(downsample=6, in_channels=1, out_channels=3),
    ('reshading', 'normal'): lambda : UNet(downsample=4, in_channels=3, out_channels=3),
    ('sobel_edges', 'principal_curvature'): lambda : UNet(downsample=5, in_channels=1, out_channels=3),
    ('depth_zbuffer', 'principal_curvature'): lambda : UNet(downsample=4, in_channels=1, out_channels=3),
    ('principal_curvature', 'depth_zbuffer'): lambda : UNet(downsample=6, in_channels=3, out_channels=1),
    ('rgb', 'normal'): lambda : UNet(downsample=6),
    ('rgb', 'keypoints2d'): lambda : UNet(downsample=3, out_channels=1),
}

def get_model(src_task, dest_task):

    if isinstance(src_task, str) and isinstance(dest_task, str):
        src_task, dest_task = get_task(src_task), get_task(dest_task)

    if (src_task.name, dest_task.name) in model_types:
        return model_types[(src_task.name, dest_task.name)]()

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, ImageTask):
        return UNet(downsample=5, in_channels=src_task.shape[0], out_channels=dest_task.shape[0])

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, ClassTask):
        return ResNet(in_channels=src_task.shape[0], out_channels=dest_task.classes)

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, PointInfoTask):
        return ResNet(out_channels=dest_task.out_channels)

    return None



"""
Abstract task type definitions.
Includes Task, ImageTask, ClassTask, PointInfoTask, and SegmentationTask.
"""

class Task(object):
    variances = {
        "normal": 1.0,
        "principal_curvature": 1.0,
        "sobel_edges": 5,
        "depth_zbuffer": 0.1,
        "reshading": 1.0,
        "keypoints2d": 0.3,
        "keypoints3d": 0.6,
        "edge_occlusion": 0.1,
    }
    """ General task output space"""
    def __init__(self, name,
            file_name=None, file_name_alt=None, file_ext="png", file_loader=None,
            plot_func=None
        ):

        super().__init__()
        self.name = name
        self.file_name, self.file_ext = file_name or name, file_ext
        self.file_name_alt = file_name_alt or self.file_name
        self.file_loader = file_loader or self.file_loader
        self.plot_func = plot_func or self.plot_func
        self.variance = Task.variances.get(name, 1.0)
        self.kind = name

    def norm(self, pred, target, batch_mean=True, compute_mse=True):
        if batch_mean:
            loss = ((pred - target)**2).mean() if compute_mse else ((pred - target).abs()).mean()
        else:
            loss = ((pred - target)**2).mean(dim=1).mean(dim=1).mean(dim=1) if compute_mse \
                    else ((pred - target).abs()).mean(dim=1).mean(dim=1).mean(dim=1)

        return loss, (loss.mean().detach(),)


    def nll(self, pred, target, batch_mean=True, mask=None):
        nchannels = pred.size(1)//2
        mux, sigma = pred[:,:nchannels], pred[:, nchannels:].exp()+1e-10
        lap_dist = torch.distributions.Laplace(loc=mux, scale=sigma)
        logprobs = -lap_dist.log_prob(target)

        if mask is not None: nll = logprobs*mask
        if batch_mean:
            nll = (nll).mean()
        else:
            nll = (nll).mean(dim=1).mean(dim=1).mean(dim=1)
        return nll, (nll.detach(),)

    def __call__(self, size=256):
        task = copy.deepcopy(self)
        return task

    def plot_func(self, data, name, logger, **kwargs):
        ### Non-image tasks cannot be easily plotted, default to nothing
        pass

    def file_loader(self, path, resize=None, seed=0, T=0):
        raise NotImplementedError()

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


"""
Abstract task type definitions.
Includes Task, ImageTask, ClassTask, PointInfoTask, and SegmentationTask.
"""

class RealityTask(Task):
    """ General task output space"""

    def __init__(self, name, dataset, tasks=None, use_dataset=True, shuffle=False, batch_size=64):

        super().__init__(name=name)
        self.tasks = tasks if tasks is not None else \
            (dataset.dataset.tasks if hasattr(dataset, "dataset") else dataset.tasks)
        self.shape = (1,)
        if not use_dataset: return
        self.dataset, self.shuffle, self.batch_size = dataset, shuffle, batch_size
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=24, shuffle=self.shuffle, pin_memory=True
        )
        self.generator = cycle(loader)
        self.step()
        self.static = False

    @classmethod
    def from_dataloader(cls, name, loader, tasks):
        reality = cls(name, None, tasks, use_dataset=False)
        reality.loader = loader
        reality.generator = cycle(loader)
        reality.static = False
        reality.step()
        return reality

    @classmethod
    def from_static(cls, name, data, tasks):
        reality = cls(name, None, tasks, use_dataset=False)
        reality.task_data = {task: x.requires_grad_() for task, x in zip(tasks, data)}
        reality.static = True
        return reality

    def norm(self, pred, target, batch_mean=True):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

    def step(self):
        self.task_data = {task: x.requires_grad_() for task, x in zip(self.tasks, next(self.generator))}

    def reload(self):
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=24, shuffle=self.shuffle, pin_memory=True
        )
        self.generator = cycle(loader)

class ImageTask(Task):
    """ Output space for image-style tasks """

    def __init__(self, *args, **kwargs):

        self.shape = kwargs.pop("shape", (3, 256, 256))
        self.mask_val = kwargs.pop("mask_val", -1.0)
        self.transform = kwargs.pop("transform", lambda x: x)
        self.resize = kwargs.pop("resize", self.shape[1])
        self.blur_radius = None
        self.image_transform = self.load_image_transform()
        super().__init__(*args, **kwargs)

    @staticmethod
    def build_mask(target, val=0.0, tol=1e-3):
        if target.shape[1] == 1:
            mask = ((target >= val - tol) & (target <= val + tol))
            mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
            return (~mask).expand_as(target)

        mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
        mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
        mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
        mask = (mask1 & mask2 & mask3).unsqueeze(1)
        mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
        return (~mask).expand_as(target)

    def norm(self, pred, target, batch_mean=True, compute_mask=0, compute_mse=True, mask=None):
        if compute_mask:
            if mask is None: mask = ImageTask.build_mask(target, val=self.mask_val).float()
            return super().norm(pred*mask.float(), target*mask.float(), batch_mean=batch_mean, compute_mse=compute_mse)
        else:
            return super().norm(pred, target, batch_mean=batch_mean, compute_mse=compute_mse)

    def nll(self, pred, target, batch_mean=True, compute_mask=0, mask=None):
        if compute_mask:
            if mask is None: mask = ImageTask.build_mask(target, val=self.mask_val).float()
            return super().nll(pred, target, batch_mean=batch_mean, mask=mask)
        else:
            return super().nll(pred, target, batch_mean=batch_mean)

    def __call__(self, size=256, blur_radius=None):
        task = copy.deepcopy(self)
        task.shape = (3, size, size)
        task.resize = size
        task.blur_radius = blur_radius
        task.name +=  "blur" if blur_radius else str(size)
        task.base = self
        return task

    def plot_func(self, data, name, logger, resize=None, nrow=2):
        logger.images(data.clamp(min=0, max=1), name, nrow=nrow, resize=resize or self.resize)

    def file_loader(self, path, resize=None, crop=None, seed=0, jitter=False, blur_radius=None, noise=None, jpeg=None, val_distortion_name=None, val_severity=None):
        image_transform = self.load_image_transform(resize=resize, crop=crop, seed=seed, jitter=jitter, blur_radius=blur_radius, noise=noise, jpeg=jpeg, val_distortion_name=val_distortion_name, val_severity=val_severity)
        return image_transform(Image.open(open(path, 'rb')))[0:3]

    def load_image_transform(self, resize=None, crop=None, seed=0, jitter=False, blur_radius=None, noise=None, jpeg=None, val_distortion_name=None, val_severity=None):
        size = resize or self.resize
        random.seed(seed)
        jitter_transform = lambda x: x
        if jitter: jitter_transform = transforms.ColorJitter(0.4,0.4,0.4,0.1)
        crop_transform = lambda x: x
        if crop is not None: crop_transform = transforms.CenterCrop(crop)
        blur = [partial(gaussian_blur,c=blur_radius)] if blur_radius else []
        noise = [partial(gaussian_noise,c=noise)] if noise else []
        jpeg = [partial(jpeg_compression,c=jpeg)] if jpeg else []

        val_dist = [partial(eval(val_distortion_name),severity=val_severity)] if val_severity is not None else []

        return transforms.Compose(blur+noise+jpeg+val_dist+[
            crop_transform,
            transforms.Resize(size, interpolation=PIL.Image.BILINEAR),
            jitter_transform,
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            self.transform]
        )

class ImageClassTask(ImageTask):
    """ Output space for image-class segmentation tasks """

    def __init__(self, *args, **kwargs):

        self.classes = kwargs.pop("classes", (3, 256, 256))
        super().__init__(*args, **kwargs)

    def norm(self, pred, target):
        loss = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(target, dim=1))
        return loss, (loss.detach(),)

    def plot_func(self, data, name, logger, resize=None):
        _, idx = torch.max(data, dim=1)
        idx = idx.float()/16.0
        idx = idx.unsqueeze(1).expand(-1, 3, -1, -1)
        logger.images(idx.clamp(min=0, max=1), name, nrow=2, resize=resize or self.resize)

    def file_loader(self, path, resize=None):

        data = (self.image_transform(Image.open(open(path, 'rb')))*255.0).long()
        one_hot = torch.zeros((self.classes, data.shape[1], data.shape[2]))
        one_hot = one_hot.scatter_(0, data, 1)
        return one_hot


class PointInfoTask(Task):
    """ Output space for point-info prediction tasks (what models do we evem use?) """

    def __init__(self, *args, **kwargs):

        self.point_type = kwargs.pop("point_type", "vanishing_points_gaussian_sphere")
        self.out_channels = 9
        super().__init__(*args, **kwargs)

    def plot_func(self, data, name, logger):
        logger.window(name, logger.visdom.text, str(data.data.cpu().numpy()))

    def file_loader(self, path, resize=None):
        points = json.load(open(path))[self.point_type]
        return np.array(points['x'] + points['y'] + points['z'])




"""
Current list of task definitions.
Accessible via tasks.{TASK_NAME} or get_task("{TASK_NAME}")
"""

##### benchmark corruptions #####
def gaussian_noise(x, c=0.0):
    # c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    return Image.fromarray(np.uint8(x))

def gaussian_blur(x, c=0.0):
    # c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    x = np.clip(x, 0, 1) * 255

    return Image.fromarray(np.uint8(x))

def jpeg_compression(x, c=0.0):
    # c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    output.seek(0)
    # x = PILImage.open(output)

    return Image.open(output)
###############################


def clamp_maximum_transform(x, max_val=8000.0):
    x = x.unsqueeze(0).float() / max_val
    return x[0].clamp(min=0, max=1)

def crop_transform(x, max_val=8000.0):
    x = x.unsqueeze(0).float() / max_val
    return x[0].clamp(min=0, max=1)

def sobel_transform(x):
    image = x.data.cpu().numpy().mean(axis=0)
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    sx = ndimage.sobel(blur, axis=0, mode='constant')
    sy = ndimage.sobel(blur, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    edge = torch.FloatTensor(sob).unsqueeze(0)
    return edge

def blur_transform(x, max_val=4000.0):
    if x.shape[0] == 1:
        x = x.squeeze(0)
    image = x.data.cpu().numpy()
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    norm = torch.FloatTensor(blur).unsqueeze(0)**0.8 / (max_val**0.8)
    norm = norm.clamp(min=0, max=1)
    if norm.shape[0] != 1:
        norm = norm.unsqueeze(0)
    return norm

def binarized_transform(x):
    image = (x>0.5)*1.0
    return image.float()

def laplace_transform(x):
    image = x.data.cpu().numpy().mean(axis=0)
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    lap = ndimage.laplace(blur)
    edge = torch.FloatTensor(lap).unsqueeze(0)
    return edge

def gauss_transform(x):
    x_cpu = x.data.cpu().numpy()
    r, g, b = x_cpu[0,:], x_cpu[1,:], x_cpu[2,:]
    fr, fg, fb = ndimage.filters.gaussian_filter(r, sigma=4), ndimage.filters.gaussian_filter(g, sigma=4), ndimage.filters.gaussian_filter(b, sigma=4)
    fr, fg, fb = fr[None,:], fg[None,:], fb[None,:]
    x_f = np.concatenate( (fr,fg,fb), axis=0)
    image = torch.FloatTensor(x_f)
    return image

def emboss_transform(x):
    x = x.mean(0,keepdim=True)
    image = transforms.ToPILImage()(x.cpu())
    imageEmboss = image.filter(ImageFilter.EMBOSS)
    image = transforms.ToTensor()(imageEmboss)
    return 

def grey_transform(x):
    return x.mean(0,keepdim=True)


from pytorch_wavelets import DWTForward, DWTInverse

xfm = DWTForward(J=3, mode='zero', wave='db1').cuda()
    
def wav_transform(x):
    x_h, x_l = xfm(x.unsqueeze(0))
    x_h = F.interpolate(x_h, size=256, mode='bilinear')
    x_l_0, x_l_1, x_l_2 = F.interpolate(x_l[0][:,:,0,:], size=256, mode='bilinear'), F.interpolate(x_l[1][:,:,0,:], size=256, mode='bilinear') , F.interpolate(x_l[2][:,:,0,:], size=256, mode='bilinear')
    x_final = torch.cat((x_h.squeeze(),x_l_0.squeeze(),x_l_1.squeeze(),x_l_2.squeeze()), dim=0)

    return x_final

def emboss4d_transform(x):
    x = x.mean(0,keepdim=True)
    x = (x*255).round().unsqueeze(0)
    
    image1, image2, image3, image4 = F.conv2d(x,emboss_weights,padding=1),F.conv2d(x,emboss_weights_2,padding=1),F.conv2d(x,emboss_weights_3,padding=1),F.conv2d(x,emboss_weights_4,padding=1)
    image1, image2, image3, image4 = image1 + 128.0, image2 + 128.0, image3 + 128.0, image4 + 128.0
    image1, image2, image3, image4 = image1.clamp(min=0.0,max=255.0), image2.clamp(min=0.0,max=255.0), image3.clamp(min=0.0,max=255.0), image4.clamp(min=0.0,max=255.0)
    image1, image2, image3, image4 = image1 / 255.0, image2 / 255.0, image3 / 255.0, image4 / 255.0

    image = torch.cat((image1,image2,image3,image4), dim=1)
    return image.squeeze(0)

def sharp_transform(x):
    x_cpu = x.data.cpu().numpy()
    r, g, b = x_cpu[0,:], x_cpu[1,:], x_cpu[2,:]
    #fr, fg, fb = np.fft.fft2(r), np.fft.fft2(g), np.fft.fft2(b)
    #fr, fg, fb = ndimage.fourier_gaussian(fr, sigma=4), ndimage.fourier_gaussian(fg, sigma=4), ndimage.fourier_gaussian(fb, sigma=4)
    fr, fg, fb = ndimage.filters.gaussian_filter(r, sigma=4), ndimage.filters.gaussian_filter(g, sigma=4), ndimage.filters.gaussian_filter(b, sigma=4)
    
    fr, fg, fb = fr[None,:], fg[None,:], fb[None,:]
    x_f = np.concatenate( (fr,fg,fb), axis=0)
    
    x_f = (x_cpu - x_f) + x_cpu
    image = torch.FloatTensor(x_f).clamp(min=0.0,max=1.0)
    #result = x_f
    #result_rf, result_gf, result_bf = result[0,:], result[1,:], result[2,:]
    #result_r, result_g, result_b = np.fft.ifft2(result_rf).real, np.fft.ifft2(result_gf).real, np.fft.ifft2(result_bf).real
    #result_r, result_g, result_b = result_r[None,:], result_g[None,:], result_b[None,:]
    #image = np.concatenate( (result_r,result_g,result_b), axis=0)
    #image = torch.FloatTensor(image)
    
    return image

def get_task(task_name):
    return task_map[task_name]



tasks = [
    ImageTask('rgb'),
    ImageTask('imagenet', mask_val=0.0),
    ImageTask('normal', mask_val=0.502),
    ImageTask('principal_curvature', mask_val=0.0),
    ImageTask('depth_zbuffer',
        shape=(1, 256, 256),
        mask_val=1.0,
        transform=partial(clamp_maximum_transform, max_val=8000.0),
    ),
    ImageClassTask('segment_semantic',
        file_name_alt="segmentsemantic",
        shape=(16, 256, 256), classes=16,
    ),
    ImageTask('reshading', mask_val=0.0507),
    ImageTask('stackedr', mask_val=0.0507),
    ImageTask('edge_occlusion',
        shape=(1, 256, 256),
        transform=partial(blur_transform, max_val=4000.0),
    ),
    ImageTask('sobel_edges',
        shape=(1, 256, 256),
        file_name='rgb',
        transform=sobel_transform,
    ),
    ImageTask('keypoints3d',
        shape=(1, 256, 256),
        transform=partial(clamp_maximum_transform, max_val=64131.0),
    ),
    ImageTask('keypoints2d',
        shape=(1, 256, 256),
        transform=partial(blur_transform, max_val=2000.0),
    ),
    ImageTask('grey',
        shape=(1, 256, 256),
        transform=partial(grey_transform),
    ),
    ImageTask('laplace_edges',
        shape=(1, 256, 256),
        transform=partial(laplace_transform),
    ),
    ImageTask('keypnt',
        shape=(1, 256, 256),
        file_name='keypoints2d',
        #transform=partial(keypnt_transform),
        transform=partial(blur_transform, max_val=2000.0),
    ),
    # ImageTask('superpix',
    #     shape=(3, 256, 256),
    #     transform=partial(superpix_transform),
    # ),
    # ImageTask('otsubin',
    #     shape=(1, 256, 256),
    #     transform=partial(otsubin_transform),
    # ),
    ImageTask('emboss4d',
        shape=(4, 256, 256),
        transform=partial(emboss4d_transform),
    ),
    ImageTask('gauss',
        shape=(3, 256, 256),
        file_name='rgb',
        transform=gauss_transform,
    ),
    ImageTask('sharp',
        shape=(3, 256, 256),
        file_name='rgb',
        transform=sharp_transform,
    ),
    ImageTask('emboss',
        shape=(1, 256, 256),
        file_name='rgb',
        transform=emboss_transform,
    ),
    ImageTask('wav',
        shape=(12, 256, 256),
        file_name='rgb',
        transform=wav_transform,
    ),

    ImageTask('normal_t0', mask_val=0.502),
    ImageTask('depth_zbuffer_t0',
        shape=(1, 256, 256),
        mask_val=1.0,
        transform=partial(clamp_maximum_transform, max_val=8000.0),
    ),
    ImageTask('reshading_t0', mask_val=0.0507),

]


task_map = {task.name: task for task in tasks}
tasks = namedtuple('TaskMap', task_map.keys())(**task_map)


if __name__ == "__main__":
    IPython.embed()
