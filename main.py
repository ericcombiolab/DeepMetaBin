"""
---------------------------------------------------------------------
-- Author: RAO Mingxing
---------------------------------------------------------------------

Main file to execute the model on the MNIST dataset

"""
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data
from model.GMVAE import *
import _pickle as pickle 

#########################################################
## Input Parameters
#########################################################

parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepMetaBin Clustering')

## Dataset
parser.add_argument('--train_set', type=str, default='data/sharon/training_set.pkl', help='training set (.pkl)')
parser.add_argument('--test_set', type=str, default='data/sharon/test_set.pkl', help='test set (.pkl) (they are the same without the assembly graph)')
parser.add_argument('--output_dir', type=str, default='data/sharon/', help='test set (.pkl) (they are the same without the assembly graph)')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')

## GPU
parser.add_argument('--cuda', type=int, default=0,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')

## Training
parser.add_argument('--epochs', type=int, default=300,
                    help='number of total epochs to run (default: 200)')
parser.add_argument('--batch_size', default=129, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--batch_size_val', default=200, type=int,
                    help='mini-batch size of validation (default: 200)')
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='learning rate (default: 0.001)')
parser.add_argument('--decay_epoch', default=-1, type=int, 
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')

## Architecture
parser.add_argument('--num_classes', type=int, default=11,
                    help='number of classes (default: 10)')
parser.add_argument('--gaussian_size', default=32, type=int,
                    help='gaussian size (default: 64)')
parser.add_argument('--input_size', default=104, type=int,
                    help='input size (default: 784)')


## Gumbel parameters
parser.add_argument('--init_temp', default=0.6, type=float,
                    help='Initial temperature used in gumbel-softmax (recommended 0.5-1.0, default:1.0)')
parser.add_argument('--decay_temp', default=1, type=int, 
                    help='Set 1 to decay gumbel temperature at every epoch (default: 1)')
parser.add_argument('--hard_gumbel', default=1, type=int, 
                    help='Set 1 to use the hard version of gumbel-softmax (default: 1)')
parser.add_argument('--min_temp', default=0.5, type=float, 
                    help='Minimum temperature of gumbel-softmax after annealing (default: 0.5)' )
parser.add_argument('--decay_temp_rate', default=0.013862944, type=float,
                    help='Temperature decay rate at every epoch (default: 0.013862944)')

## Loss function parameters
parser.add_argument('--w_gauss', default=1/(32*200), type=float,
                    help='weight of gaussian loss (default: 1)')
parser.add_argument('--w_categ', default=1/(32*200), type=float,
                    help='weight of categorical loss (default: 1)')
parser.add_argument('--w_rec', default=1, type=float,
                    help='weight of reconstruction loss (default: 1)')
parser.add_argument('--rec_type', type=str, choices=['bce', 'mse'],
                    default='bce', help='desired reconstruction loss function (default: bce)')

## Others
parser.add_argument('--verbose', default=1, type=int,
                    help='print extra information at every epoch.(default: 0)')
parser.add_argument('--random_search_it', type=int, default=20,
                    help='iterations of random search (default: 20)')
parser.add_argument('--truth', type=str, default='../DeepBin/data/sharon/labels.csv',
                    help='ground truth of your test data')
parser.add_argument('--cutoff', type=int, default=1000,
                    help='abandon those contig with length smaller than a cutoff value')

args = parser.parse_args()

if args.cuda == 1:
   os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

## Random Seed
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if args.cuda:
  torch.cuda.manual_seed(SEED)

#########################################################
## Read Data
#########################################################
if args.cuda == 0:
  train_dataset = torch.load(args.train_set, map_location ='cpu')
  test_dataset = torch.load(args.test_set, map_location ='cpu')
else:
  train_dataset = torch.load(args.train_set, map_location='cuda')
  test_dataset = torch.load(args.train_set, map_location='cuda')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
val_loader = test_loader


#########################################################
## Train and Test Model
#########################################################
gmvae = GMVAE(args)

## Training Phase
best_result, best_metric_results = gmvae.train(train_loader, val_loader, test_loader)
precision, recall, f1_score, ari = best_metric_results
contignames, predicts = best_result

# Testing Phase
# precision, recall, f1_score, ari = gmvae.test(test_loader, output_result = True)

print("Valid - Precision: {:.5f}; Recall: {:.5f}; F1_score: {:.5f}; ARI: {:.5f}".format(precision, recall, f1_score, ari))

with open(os.path.join(args.output_dir, 'result.csv'), 'w') as f:
    for contig, cluster in zip(contignames, predicts):
      f.write(f'{contig},{cluster}\n')


