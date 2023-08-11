from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import logging
import glob
import pandas as pd
import scipy.misc
from matplotlib import pyplot as plt

import torch

from torchvision.utils import save_image

# from data.CamVid_loader import CamVidDataset
# from data.utils import decode_segmap, decode_seg_map_sequence
from mypath import Path
from utils.metrics import Evaluator
from data import make_data_loader

from model.FPN import FPN
from model.resnet import resnet

#from torchsummary import summary


def get_cityscapes_labels():
    return np.array([
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def decode_seg_map_sequence(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'Cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[1], label_mask.shape[2], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    # if plot:
    #     plt.imshow(rgb)
    #     plt.show()
    # else:
    return rgb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a FPN Semantic Segmentation network')
    parser.add_argument('--dataset', dest='dataset',
					    help='training dataset',
					    default='Cityscapes', type=str)
    parser.add_argument('--net', dest='net',
					    help='resnet101, res152, etc',
					    default='resnet101', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
					    help='starting epoch',
					    default=1, type=int)
    parser.add_argument('--epochs', dest='epochs',
					    help='number of iterations to train',
					    default=2000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
					    help='directory to save models',
					    default="D:\\disk\\midterm\\experiment\\code\\semantic\\fpn\\fpn\\run",
					    type=str)
    parser.add_argument('--num_workers', dest='num_workers',
					    help='number of worker to load data',
					    default=0, type=int)
    # cuda
    parser.add_argument('--cuda', dest='cuda',
					    help='whether use multiple GPUs',
                        default=True,
					    action='store_true')
    # batch size
    parser.add_argument('--bs', dest='batch_size',
					    help='batch_size',
					    default=1, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
					    help='training optimizer',
					    default='sgd', type=str)
    parser.add_argument('--lr', dest='lr',
					    help='starting learning rate',
					    default=0.001, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
					    help='step to do learning rate decay, uint is epoch',
					    default=500, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
					    help='learning rate decay ratio',
					    default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
					    help='training session',
					    default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
					    help='resume checkpoint or not',
					    default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
					    help='checksession to load model',
					    default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
					    help='checkepoch to load model',
					    default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
					    help='checkpoint to load model',
					    default=0, type=int)

    # log and display
    parser.add_argument('--use_tfboard', dest='use_tfboard',
					    help='whether use tensorflow tensorboard',
					    default=True, type=bool)

    # configure validation
    parser.add_argument('--no_val', dest='no_val',
                        help='not do validation',
                        default=False, type=bool)
    parser.add_argument('--eval_interval', dest='eval_interval',
                        help='iterval to do evaluate',
                        default=2, type=int)

    parser.add_argument('--checkname', dest='checkname',
                        help='checkname',
                        default=None, type=str)

    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')

    # test confit
    parser.add_argument('--plot', dest='plot',
                        help='wether plot test result image',
                        default=False, type=bool)
    parser.add_argument('--exp_dir', dest='experiment_dir',
                          help='dir of experiment',
                          type=str)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.dataset == 'CamVid':
        num_class = 32
    elif args.dataset == 'Cityscapes':
        num_class = 19

    if args.net == 'resnet101':
        blocks = [2, 4, 23, 3]
        model = FPN(blocks, num_class, back_bone=args.net)
        #summary(model, (1, 3, 512, 512))

    if args.checkname is None:
        args.checkname = 'fpn-' + str(args.net)

    for index in range(1, 61):       # to evaluate all the epochs of training (60)
        evaluator = Evaluator(num_class)
    
        # Trained model path and name
        experiment_dir = args.experiment_dir
        load_name = args.experiment_dir+"/Cityscapes/fpn-resnet101/experiment_0/checkpoint_{}.pth.tar".format(index)
        # Load trained model
        if not os.path.isfile(load_name):
            raise RuntimeError("=> no checkpoint found at '{}'".format(load_name))
        print('====>loading trained model from ' + load_name)
        checkpoint = torch.load(load_name)
        checkepoch = checkpoint['epoch']
        
        if args.cuda:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
            
        # Load image and save in test_imgs
        test_imgs = []
        test_label = []
        if args.dataset == "CamVid":
            root_dir = Path.db_root_dir('CamVid')
            test_file = os.path.join(root_dir, "val.csv")
            test_data = CamVidDataset(csv_file=test_file, phase='val')
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
        elif args.dataset == "Cityscapes":
            kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
            #_, test_loader, _, _ = make_data_loader(args, **kwargs)
            #_, val_loader, test_loader, _ = make_data_loader(args, **kwargs)
            _, val_loader, test_loader, _ = make_data_loader(args, **kwargs)
        else:
            raise RuntimeError("dataset {} not found.".format(args.dataset))
    
        # test
        Acc = []
        Acc_class = []
        mIoU = []
        FWIoU = []
        results = []
        idx=0
        for iter, batch in enumerate(val_loader):
            if args.dataset == 'CamVid':
                image, target = batch['X'], batch['l']
            elif args.dataset == 'Cityscapes':
                image, target = batch['image'], batch['label']
                    
            else:
                raise NotImplementedError
    
            if args.cuda:
                image, target, model = image.cuda(), target.cuda(), model.cuda()
                
            with torch.no_grad():
                output = model(image)
            pred = output.data.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            target = target.cpu().numpy()
            evaluator.add_batch(target, pred)
    
            # show result
            pred_rgb = decode_seg_map_sequence(pred, args.dataset, args.plot)
            idx +=1
                
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    
        print('Mean evaluate result on dataset {}, ckpt {}'.format(args.dataset, index))
        print('Acc:{:.5f}\tAcc_class:{:.5f}\tmIoU:{:.5f}\tFWIoU:{:.5f}'.format(Acc, Acc_class, mIoU, FWIoU))
               

if __name__ == "__main__":
    main()
