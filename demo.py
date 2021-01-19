import torch
from torchvision import transforms

from modules.unet import UNet, UNetReshade

import PIL
from PIL import Image

import argparse
import os.path
from pathlib import Path
import glob, math
import sys
from functools import partial

import pdb
from models import TrainableModel, WrapperModel, DataParallelModel
from utils import *
from distortions import *



parser = argparse.ArgumentParser(description='Visualize output for a single Task')

parser.add_argument('--task', dest='task', help="normal, depth or reshading")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

parser.add_argument('--distortion', dest='distortionname', help="name of the distortion to be applied")
parser.set_defaults(store_name='None')

parser.add_argument('--severity', dest='severity', help="severity of the distortion from 1 to 5")
parser.set_defaults(store_name='None')

args = parser.parse_args()

root_dir = './models/'

#get distortion name and severity
distortions = [None, 'shot_noise','speckle_noise','impulse_noise','defocus_blur','contrast','brightness','saturate','jpeg_compression','pixelate','spatter','glass_blur', 'gaussian_noise', 'gaussian_blur']
try:
    distortion_index = distortions.index(args.distortionname)
    distortion = distortions[distortion_index]
except:
    print("distortion should be one of the following: 'shot_noise','speckle_noise','impulse_noise','defocus_blur','contrast','brightness','saturate','jpeg_compression','pixelate','spatter','glass_blur', 'gaussian_noise', 'gaussian_blur'")
    sys.exit()

severities = [None, '1', '2', '3', '4', '5']
try:
    severity_index = severities.index(args.severity)
    if severity_index is not 0:
        severity = int(severities[severity_index])
except:
    print("severity should be one of the following: 1, 2, 3, 4, 5")
    sys.exit()



if distortion is not None:
    noise = [partial(eval(distortion),severity=severity)] if severity is not None else []
    trans_totensor = transforms.Compose(noise+[transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
else:
    trans_totensor = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.BILINEAR),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor()])
trans_topil = transforms.ToPILImage()


# get target task and model
target_tasks = ['normal','depth_zbuffer','reshading']
try:
    task_index = target_tasks.index(args.task)
    target_task = target_tasks[task_index]
except:
    print("task should be one of the following: normal, depth_zbuffer, reshading")
    sys.exit()
#models = [UNet(), UNet(downsample=6, out_channels=1), UNetReshade(downsample=5)]

#direct, emboss4d, grey, laplace, gauss, sobel, wav, sharp
models_normal = [UNet(out_channels=6), UNet(in_channels=4, downsample=6, out_channels=6), UNet(downsample=6, in_channels=1, out_channels=6), UNet(in_channels=1, downsample=6, out_channels=6), UNet(in_channels=3, downsample=6, out_channels=6), UNet(downsample=5, in_channels=1, out_channels=6), UNet(downsample=6, in_channels=12, out_channels=6), UNet(in_channels=3, downsample=6, out_channels=6)]
models_reshading = [UNet(downsample=5, out_channels=2), UNet(in_channels=4, downsample=5, out_channels=2), UNet(downsample=5, in_channels=1, out_channels=2), UNet(in_channels=1, downsample=5, out_channels=2), UNet(in_channels=3, downsample=5, out_channels=2), UNet(downsample=5, in_channels=1, out_channels=2), UNet(downsample=5, in_channels=12, out_channels=2), UNet(in_channels=3, downsample=5, out_channels=2)]
models_depth_zbuffer = [UNet(downsample=6, out_channels=2), UNet(in_channels=4, downsample=6, out_channels=2), UNet(downsample=6, in_channels=1, out_channels=2), UNet(in_channels=1, downsample=6, out_channels=2), UNet(in_channels=3, downsample=6, out_channels=2), UNet(downsample=6, in_channels=1, out_channels=2), UNet(downsample=5, in_channels=12, out_channels=2), UNet(in_channels=3, downsample=6, out_channels=2)]
models_normal_deepens = UNet(out_channels=6)
models_reshading_deepens = UNet(downsample=5, out_channels=2)
models_depth_zbuffer_deepens = UNet(downsample=6, out_channels=2)
#model = models[task_index]

if target_task is 'normal':
    path = root_dir + "xd_ens_normal.pth"
    models = models_normal
    path_deepens = root_dir + "deep_ens_normal_cons.pth"
    models_deepens = models_normal_deepens
if target_task is 'reshading':
    path = root_dir + "xd_ens_reshading.pth"
    models = models_reshading
    path_deepens = root_dir + "deep_ens_reshading_cons.pth"
    models_deepens = models_reshading_deepens
if target_task is 'depth_zbuffer':
    path = root_dir + "xd_ens_depth_zbuffer.pth"
    models = models_depth_zbuffer
    path_deepens = root_dir + "deep_ens_depth_zbuffer_cons.pth"
    models_deepens = models_depth_zbuffer_deepens

map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')


def save_outputs(img_path, output_file_name):

    img = Image.open(img_path)
    img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(DEVICE)

    if distortion is not None:
        trans_topil(img_tensor[0].clamp(min=0, max=1).cpu()).save(args.output_path+'/'+'distorted_input.png')

    # compute baseline and consistency output
    #for type in ['baseline','consistency']:
    #    path = root_dir + 'rgb2'+args.task+'_'+type+'.pth'
    #    model_state_dict = torch.load(path, map_location=map_location)
    #    model.load_state_dict(model_state_dict)
    #    baseline_output = model(img_tensor).clamp(min=0, max=1)
    #    trans_topil(baseline_output[0]).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+type+'.png')
    
    # compute all 8 path outputs
    #pdb.set_trace()
    all_models_state_dict = torch.load(path, map_location=map_location)

    direct_model = WrapperModel(DataParallelModel(models[0].to(DEVICE)))
    #pdb.set_trace()
    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"')"])
    direct_output = direct_model(img_tensor)#.clamp(min=0, max=1)

    emboss_model = WrapperModel(DataParallelModel(models[1].to(DEVICE)))
    emboss_model.load_state_dict(all_models_state_dict["('emboss4d', '"+target_task+"')"])
    emboss_output = emboss_model(emboss4d_kernel(img_tensor))#.clamp(min=0, max=1)

    grey_model = WrapperModel(DataParallelModel(models[2].to(DEVICE)))
    grey_model.load_state_dict(all_models_state_dict["('grey', '"+target_task+"')"])
    grey_output = grey_model(greyscale(img_tensor))#.clamp(min=0, max=1)

    laplace_model = WrapperModel(DataParallelModel(models[3].to(DEVICE)))
    laplace_model.load_state_dict(all_models_state_dict["('laplace_edges', '"+target_task+"')"])
    laplace_output = laplace_model(laplace_kernel(img_tensor))#.clamp(min=0, max=1)

    gauss_model = WrapperModel(DataParallelModel(models[4].to(DEVICE)))
    gauss_model.load_state_dict(all_models_state_dict["('gauss', '"+target_task+"')"])
    gauss_output = gauss_model(gauss_kernel(img_tensor))#.clamp(min=0, max=1)

    sobel_model = WrapperModel(DataParallelModel(models[5].to(DEVICE)))
    sobel_model.load_state_dict(all_models_state_dict["('sobel_edges', '"+target_task+"')"])
    sobel_output = sobel_model(sobel_kernel(img_tensor))#.clamp(min=0, max=1)

    wav_model = WrapperModel(DataParallelModel(models[6].to(DEVICE)))
    wav_model.load_state_dict(all_models_state_dict["('wav', '"+target_task+"')"])
    wav_output = wav_model(wav_kernel(img_tensor))#.clamp(min=0, max=1)

    sharp_model = WrapperModel(DataParallelModel(models[7].to(DEVICE)))
    sharp_model.load_state_dict(all_models_state_dict["('sharp', '"+target_task+"')"])
    sharp_output = sharp_model(sharp_kernel(img_tensor))#.clamp(min=0, max=1)

    #merged_outputs = torch.Tensor().cuda()
    merged_outputs = torch.cat((direct_output, emboss_output, grey_output, laplace_output, gauss_output, sobel_output, wav_output, sharp_output),dim=1)

    npaths = 8
    di_ind = np.diag_indices(npaths)
    nchannels = int(merged_outputs.size(1)//(npaths*2))
    inds = np.arange(npaths)*2+1   # indices of channel0 sigmas
    SQRT2 = math.sqrt(2)
    for i in range(nchannels):
        inds_ = nchannels*inds+i
        merged_outputs[:,inds_] = merged_outputs[:,inds_].exp()*SQRT2  # convert to sigma from log(b)

    ######## get sig avg weights
    if nchannels==1:
        muind=inds-1
        sigind=inds
    else:
        muind = np.array([0,1,2,6,7,8,12,13,14,18,19,20,24,25,26,30,31,32,36,37,38,42,43,44])  #  8 paths
        sigind = muind+3
    sig_avg_weights = torch.cuda.FloatTensor(merged_outputs[:,:npaths].size()).fill_(0.0)
    total_inv_sig = (1./merged_outputs[:,sigind].pow(2)).sum(1)
    for i in range(npaths):
        sig_avg_weights[:,i] = (1./ merged_outputs[:,2*i*nchannels+nchannels:2*(i+1)*nchannels].pow(2)).sum(1) / total_inv_sig

    weights = sig_avg_weights
    merged_mu = torch.cuda.FloatTensor(merged_outputs[:,:nchannels].size()).fill_(0.0)
    merged_sig = torch.cuda.FloatTensor(merged_outputs[:,:nchannels].size()).fill_(0.0)

    for i in range(nchannels):
        inds_ = i+nchannels*inds
        ## compute correl mat                   
        cov_mat = torch.cuda.FloatTensor(merged_mu.size(0),merged_mu.size(-1),merged_mu.size(-1),int(npaths),int(npaths)).fill_(0.0)
        cov_mat[:,:,:,di_ind[0],di_ind[1]] = merged_outputs[:,inds_].pow(2).permute(0,2,3,1)

        ## merge
        merged_mu[:,i] = (merged_outputs[:,inds_-nchannels] * weights).sum(dim=1)
        weights = weights.permute(0,2,3,1)
        merged_sig[:,i] = (weights.unsqueeze(-2) @ cov_mat @ weights.unsqueeze(-1)).squeeze(-1).squeeze(-1).sqrt()
        weights = weights.permute(0,3,1,2)

    if nchannels is 1:
        var_epistemic = merged_outputs[:,[0,2,4,6,8,10,12,14]].var(1,keepdim=True)
    else:
        var_epistemic_r = merged_outputs[:,[0,6,12,18,24,30,36,42]].var(1,keepdim=True)
        var_epistemic_g = merged_outputs[:,[1,7,13,19,25,31,37,43]].var(1,keepdim=True)
        var_epistemic_b = merged_outputs[:,[2,8,14,20,26,32,38,44]].var(1,keepdim=True)
        var_epistemic = torch.cat((var_epistemic_r,var_epistemic_g,var_epistemic_b), dim=1)

    baseline_mu = direct_output[:,:nchannels]
    baseline_sig = direct_output[:,nchannels:]

    trans_topil(merged_mu[0].clamp(min=0, max=1).cpu()).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+'ours_mean.png')
    #trans_topil((merged_sig[0].exp()*SQRT2).cpu()).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+'ours_var.png')
    trans_topil((var_epistemic[0].sqrt()*SQRT2).clamp(min=0, max=1).cpu()).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+'ours_sig.png')

    trans_topil(baseline_mu[0].clamp(min=0, max=1).cpu()).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+'baseline_mean.png')
    trans_topil((baseline_sig[0].exp()*SQRT2).cpu()).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+'baseline_sig.png')



    ### GET DEEP ENS RESULTS ###

    all_models_state_dict = torch.load(path_deepens, map_location=map_location)
   
    direct_model = WrapperModel(DataParallelModel(models_deepens.to(DEVICE)))

    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"1_ens')"])
    direct_output1 = direct_model(img_tensor)#.clamp(min=0, max=1)

    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"2_ens')"])
    direct_output2 = direct_model(img_tensor)#.clamp(min=0, max=1)

    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"3_ens')"])
    direct_output3 = direct_model(img_tensor)#.clamp(min=0, max=1)

    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"4_ens')"])
    direct_output4 = direct_model(img_tensor)#.clamp(min=0, max=1)

    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"5_ens')"])
    direct_output5 = direct_model(img_tensor)#.clamp(min=0, max=1)

    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"6_ens')"])
    direct_output6 = direct_model(img_tensor)#.clamp(min=0, max=1)

    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"7_ens')"])
    direct_output7 = direct_model(img_tensor)#.clamp(min=0, max=1)

    direct_model.load_state_dict(all_models_state_dict["('rgb', '"+target_task+"8_ens')"])
    direct_output8 = direct_model(img_tensor)#.clamp(min=0, max=1)

    mu_ens = 0.125 * (direct_output1[:,:nchannels] + direct_output2[:,:nchannels] + direct_output3[:,:nchannels] + direct_output4[:,:nchannels] + direct_output5[:,:nchannels]+ direct_output6[:,:nchannels]+ direct_output7[:,:nchannels]+ direct_output8[:,:nchannels])
    merged_outputs_ens = torch.cat((direct_output1, direct_output2, direct_output3, direct_output4, direct_output5, direct_output6, direct_output7, direct_output8),dim=1)
    
    if nchannels is 1:
        var_epistemic_ens = merged_outputs_ens[:,[0,2,4,6,8,10,12,14]].var(1,keepdim=True)
    else:
        var_epistemic_ens_r = merged_outputs_ens[:,[0,6,12,18,24,30,36,42]].var(1,keepdim=True)
        var_epistemic_ens_g = merged_outputs_ens[:,[1,7,13,19,25,31,37,43]].var(1,keepdim=True)
        var_epistemic_ens_b = merged_outputs_ens[:,[2,8,14,20,26,32,38,44]].var(1,keepdim=True)
        var_epistemic_ens = torch.cat((var_epistemic_ens_r,var_epistemic_ens_g,var_epistemic_ens_b), dim=1)

    trans_topil(mu_ens[0].clamp(min=0, max=1).cpu()).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+'deepens_mean.png')
    trans_topil((var_epistemic_ens[0].sqrt()*SQRT2).clamp(min=0, max=1).cpu()).save(args.output_path+'/'+output_file_name+'_'+args.task+'_'+'deepens_sig.png')



img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
else:
    print("invalid file path!")
    sys.exit()
