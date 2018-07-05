from __future__ import print_function

import argparse
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn    #For benchmarking algorithm. See- https://github.com/pytorch/pytorch/blob/1848cad10802db9fa0aa066d9de195958120d863/aten/src/ATen/native/cudnn/Conv.cpp#L486-L494
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets #for CIFAR10/100
import models

# from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

max_test_acc = 0 

def parser():
	model_names = ['alexnet','densenet','preresnet','resnet','resnext','vgg','wrn']
	ap = argparse.ArgumentParser(description='PyTorch CIFAR Training')

	# Datasets
	ap.add_argument('-d','--dataset',default='cifar10',type=str)
	ap.add_argument('-j','--workers',default=4,type=int,help='number of cores for loading data')

	# Parameters
	ap.add_argument('--epochs',default=250,type=int)
	ap.add_argument('--start-epoch',default=0,type=int,help='manual epoch number for restart')
	ap.add_argument('--train-batch',default=128,type=int)
	ap.add_argument('--test-batch',default=100,type=int)
	ap.add_argument('--lr','--learning-rate',default=0.1,type=float,help='initial learning rate')

	ap.add_argument('--learning-schedule',type=int,nargs='+',default=[150, 225],help='Decrease learning rate at these epochs.')
	ap.add_argument('--gamma',type=float,default=0.1,help='Learning rate scheduling parameter')
	ap.add_argument('--momentum', default=0.9, type=float)
	ap.add_argument('--weight-decay', '--wd', default=5e-4, type=float)

	# Training checkpoint savepaths
	ap.add_argument('-c', '--checkpoint', default='./checkpoints', type=str, help='path to save checkpoint')
	ap.add_argument('-r','--resume', default='./checkpoints', type=str, help='path to latest checkpoint')

	# Architecture
	ap.add_argument('--arch', '-a', default='resnet20',choices=model_names, help='model architecture: '+' | '.join(model_names)')
	ap.add_argument('--depth',type=int,default=29,help='Model depth.')
	ap.add_argument('--cardinality',type=int,default=8,help='Model cardinality (group).')
	ap.add_argument('--widen-factor',type=int,default=4,help='Widen factor. 4 -> 64, 8 -> 128, ...')
	ap.add_argument('--growthRate',type=int,default=12,help='Growth rate (DenseNet).')
	ap.add_argument('--compressionRate',type=int,default=2,help='Compression Rate (theta DenseNet)')

	ap.add_argument('--manualSeed', type=int, help='manual seed')
	ap.add_argument('-e','--evaluate',dest='evaluate',action='store_true',help='evaluate model on validation set')
	ap.add_argument('--gpu-id',default='0',type=str,help='id(s) for CUDA_VISIBLE_DEVICES')

	args = ap.parse_args()
	state = {k: v for k, v in args._get_kwargs()}
	return args,state
