from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torchvision.utils import save_image
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
from collections import namedtuple
import torch
import cv2

from tqdm import tqdm

# from data.utils import decode_segmap, decode_seg_map_sequence
from mypath import Path
from utils.metrics import Evaluator
from data import make_data_loader

from model.FPN import FPN
from model.resnet import resnet
means     = np.array([103.939, 116.779, 123.68]) / 255.



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
                        default="./",
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
                        default=5, type=int)

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
    # folder path
    parser.add_argument('--path_to_unlabel_set', dest='path_to_unlabel_set',
                          help='path of the folder to unlabel set',
                          type=str)
    parser.add_argument('--path_to_save', dest='path_to_save',
                          help='path of the folder to save inference (Cityscapes dataset)',
                          type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.dataset == 'Cityscapes':
        num_class = 19

    if args.net == 'resnet101':
        blocks = [2, 4, 23, 3]
        model = FPN(blocks, num_class, back_bone=args.net)

    if args.checkname is None:
        args.checkname = 'fpn-' + str(args.net)

    #evaluator = Evaluator(num_class)

    # Trained model path and name
    experiment_dir = args.experiment_dir
    load_name = os.path.join(experiment_dir, 'Cityscapes/fpn-resnet101/model_best.pth.tar')

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

    #convert trainid to testid
    labelsid = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    class_map = dict(zip(range(19), labelsid))
    
    path_img_to_infer = args.path_to_unlabel_set
    images_list = os.listdir(path_img_to_infer)
        
    path_to_save = args.path_to_save    
    path_gt_to_save = path_to_save + "/gtFine_trainvaltest/gtFine/train/fid1k6/"
    path_img_to_save = path_to_save + "/Cityscapes/leftImg8bit/train/fid1k6/"

    idx=0
    for image_name in tqdm(images_list):
        # test
        img_path = os.path.join(path_img_to_infer, image_name)
        image = scipy.misc.imread(img_path, mode='RGB')
        image = cv2.resize(image,(2048, 1024),cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(path_img_to_save, str('fid1k6_000000_{:06d}_leftImg8bit.png'.format(idx))), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                
        image = image[:,:,::-1]
        image = np.transpose(image,(2,0,1))
        image = torch.from_numpy(image.copy()).float()
        image = image.unsqueeze(0)
        if args.cuda:
            image,model = image.cuda(),model.cuda()
        with torch.no_grad():
            output = model(image)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        
        # show result
        pred_rgb = decode_seg_map_sequence(pred, args.dataset, args.plot)
        scipy.misc.imsave(os.path.join(path_gt_to_save, str('fid1k6_000000_{:06d}_gtFine_color.png'.format(idx))), pred_rgb)
        
        pred = pred.transpose(1,2,0)
                           for i in reversed(range(19)):
            index_pixel = np.where(pred[:,:,:]==i)
            pred[index_pixel] = class_map[i]
        #print('trainid converted to testid')

        cv2.imwrite(os.path.join(path_gt_to_save, str('fid1k6_000000_{:06d}_gtFine_labelIds.png'.format(idx))), pred)
        idx+=1


if __name__ == "__main__":
   main()
