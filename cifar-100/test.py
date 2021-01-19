
'''
  Name: test.py
  Desc: Executes testing of the models trained for rgb -> CIFAR-100 classes or mid domain -> CIFAR-100 classes domain with the cross entropy loss.
    Here are some options that may be specified for any model. If they have a
    default value, it is given at the end of the description in parens.
        Data: Standard test data (clean) for CIFAR-100. It is loaded in utils.py with the relevant PyTorch loaders using the function get_test_dataloader. If a middle domain is employed, the input RGBs are first transformed using the appropriate functions in utils.py.
        Testing:
            'b': The size of each batch. (16)
            'domain': Specifies training with directly from rgb input or one of the middle domains. 
            'net': Specifies model architecture to be used.
            'weights': Specifies the path for the model weights.
  Usage:
    python test.py -domain emboss -net resnet18 -weights $PATH_FOR_WEIGHTS


'''

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader, get_mean_std

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-domain', type=str, required=True, help='choose middle domain: direct, emboss, grey, sharp, lowpass')
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD = get_mean_std(args)

    cifar100_test_loader = get_test_dataloader(
        CIFAR100_TRAIN_MEAN,
        CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        domain=args.domain
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))