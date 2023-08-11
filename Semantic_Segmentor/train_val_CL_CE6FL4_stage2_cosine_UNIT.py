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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from data import make_data_loader
from mypath import Path

from utils.metrics import Evaluator
from utils.saver import Saver
from utils.loss import SegmentationLosses

from model.FPN import FPN
from model.resnet import resnet

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
					    default=61, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
					    help='directory to save models',
					    default=None,
					    nargs=argparse.REMAINDER)
    parser.add_argument('--num_workers', dest='num_workers',
					    help='number of worker to load data',
					    default=0, type=int)
    # cuda
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',default=True, type=bool)
    # multiple GPUs
    parser.add_argument('--mGPUs', dest='mGPUs', type=bool,
					              help='whether use multiple GPUs',
                        default=False,)
    parser.add_argument('--gpu_ids', dest='gpu_ids',
                        help='use which gpu to train, must be a comma-separated list of integers only (defalt=0)',
                        default='0', type=str)
    # batch size
    parser.add_argument('--batch_size', dest='batch_size',
					    help='batch_size',
					    default=None, type=int)

    # config optimization
    parser.add_argument('--o', dest='optimizer',
					    help='training optimizer, adam/sgd',
					    default='sgd', type=str)
    parser.add_argument('--lr', dest='lr',
					    help='starting learning rate',
					    default=0.01, type=float)
    parser.add_argument('--weight_decay', dest='weight_decay',
                        help='weight_decay',
                        default=1e-5, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
					    help='step to do learning rate decay, uint is epoch',
					    default=120, type=int)
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
					    default="./Semantic_Segmentor/saved_checkpoints/run_stage1_combine_CE6FL4/Cityscapes/fpn-resnet101/model_best.pth.tar", type=str)

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
                        default=1, type=int)

    parser.add_argument('--checkname', dest='checkname',
                        help='checkname',
                        default=None, type=str)

    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size')

    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, decay=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']
        current_lr = param_group['lr']
    return current_lr


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Dataloader
        if args.dataset == 'Cityscapes':
            kwargs = {'num_workers': args.num_workers, 'pin_memory': True}
            self.train_loader, self.val_loader, self.test_loader, self.num_class = make_data_loader(args, **kwargs)

        # Define network
        if args.net == 'resnet101':
            blocks = [2,4,23,3]
            fpn = FPN(blocks, self.num_class, back_bone=args.net)

        # Define Optimizer
        self.lr = self.args.lr
        if args.optimizer == 'adam':
            self.lr = self.lr * 0.1
            optimizer = torch.optim.Adam(fpn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            #self.lr = self.lr * 0.1
            optimizer = torch.optim.SGD(fpn.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1901, eta_min=0) #1901 with 7603 images

        # Define Criterion
        if args.dataset == 'Cityscapes':
            weight = None
            self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode='combine')

        self.model = fpn 

        #load trained model on cityscape ckpt
        self.checkpoint = torch.load(args.checkpoint)
        self.model.load_state_dict(self.checkpoint['state_dict'])
        print('Loaded checkpoint from STAGE 1!')


        self.optimizer = optimizer
        self.scheduler = scheduler

        # Define Evaluator
        self.evaluator = Evaluator(self.num_class)

        # multiple mGPUs
        if args.mGPUs:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()


        # Resuming checkpoint
        self.best_pred = 0.0
        self.lr_stage = [68, 93]
        self.lr_staget_ind = 0 

    def training(self, epoch):
        train_loss = 0.0
                
        self.model.train()
        num_img_tr = len(self.train_loader)
        current_lr = self.lr
        
        for iteration, batch in enumerate(self.train_loader):
            if self.args.dataset == 'Cityscapes':
                image, target = batch['image'], batch['label']
            else:
                raise NotImplementedError
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.optimizer.zero_grad()
            inputs = Variable(image)
            labels = Variable(target)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.long())
            loss_val = loss.item()
            loss.backward(torch.ones_like(loss))
            self.optimizer.step()
            train_loss += loss.item()

            if iteration % 100 == 0:
                print("Trainset: Epoch[{}]({}/{}):Loss:{:.4f}, lr={:.6f}".format(epoch, iteration, len(self.train_loader), loss.data, self.optimizer.param_groups[0]["lr"]))
            
            self.scheduler.step()

        print('[Trainset: Epoch: %d, numImages: %5d]' % (epoch, iteration * self.args.batch_size + image.data.shape[0]))
        print('Trainset: Loss: %.3f' % train_loss)
        
        #if self.args.no_val:
        if True: 
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                }, is_best, 'checkpoint_{}.pth.tar'.format(epoch))

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        val_loss = 0.0
        
        for iter, batch in enumerate(self.val_loader):
            if self.args.dataset == 'Cityscapes':
                image, target = batch['image'], batch['label']
            else:
                raise NotImplementedError
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            val_loss += loss.item()
            if iter % 100 == 0:
                print('Val Loss:%.3f' % (val_loss/(iter+1)))
                
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, iter * self.args.batch_size + image.shape[0]))
        print("Valset: Acc:{:.5f}, Acc_class:{:.5f}, mIoU:{:.5f}, fwIoU:{:.5f}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Valset: Loss: %.3f' % val_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

def main():
    args = parse_args()
    print('STAGE 2 1K6 FID Test combine loss CE 6 FL 4, cosine, stage 1 UNIT daynight.')
    if args.save_dir is None:
        args.save_dir = os.path.join(os.getcwd(), 'run')
    if args.checkname is None:
        args.checkname = 'fpn-' + str(args.net)

    if args.cuda and args.mGPUs:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of itegers only')

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.lr is None:
        lrs = {
            'cityscapes': 0.01,
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    print(args)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    print('DONE STAGE 2 1K6 FID Test combine loss CE 6 FL 4, cosine, stage 1 UNIT daynight.')

if __name__ == '__main__':
    main()
