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
parser = argparse.ArgumentParser(description='PyTorch Implementation of DGM Clustering')

## Used only in notebooks
parser.add_argument('-f', '--file',
                    help='Path for input file. First line should contain number of lines to search in')

## Dataset
parser.add_argument('--dataset', type=str, choices=['mnist'],
                    default='mnist', help='dataset (default: mnist)')
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
parser.add_argument('--num_classes', type=int, default=74,
                    help='number of classes (default: 10)')
parser.add_argument('--gaussian_size', default=32, type=int,
                    help='gaussian size (default: 64)')
parser.add_argument('--input_size', default=104, type=int,
                    help='input size (default: 784)')

## Partition parameters
parser.add_argument('--train_proportion', default=1.0, type=float,
                    help='proportion of examples to consider for training only (default: 1.0)')

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
parser.add_argument('--truth', type=str, default='../DeepBin/data/CAMI1_M1/labels.csv',
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
# if args.dataset == "mnist":
#   print("Loading mnist dataset...")
#   # Download or load downloaded MNIST dataset
#   train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transforms.ToTensor())
#   test_dataset = datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor())
# train_dataset = []
# with open("data/hlj/training_set.pkl", "rb") as f:
#   train_dataset = pickle.load(f)
# train_dataset = torch.load("data/CAMI1_M1/training_set.pkl", map_location='cpu')
# test_dataset = torch.load("data/CAMI1_M1/test_set.pkl", map_location='cpu')
if args.cuda == 0:
  train_dataset = torch.load("data/CAMI1_M1/training_set.pkl", map_location ='cpu')
  test_dataset = torch.load("data/CAMI1_M1/test_set.pkl", map_location ='cpu')
else:
  train_dataset = torch.load("data/CAMI1_M1/test_set.pkl", map_location='cuda')
  test_dataset = torch.load("data/CAMI1_M1/test_set.pkl", map_location='cuda')
# test_dataset = []
# with open("data/hlj/test_set.pkl", "rb") as f:
#   test_dataset = pickle.load(f)

#########################################################
## Data Partition
#########################################################
def partition_dataset(n, proportion=0.8):
  train_num = int(n * proportion)
  indices = np.random.permutation(n)
  train_indices, val_indices = indices[:train_num], indices[train_num:]
  return train_indices, val_indices

if args.train_proportion == 1.0:
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False)
  # val_loader = test_loader
  val_loader = test_loader
else:
  train_indices, val_indices = partition_dataset(len(train_dataset), args.train_proportion)
  # Create data loaders for train, validation and test datasets
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=SubsetRandomSampler(train_indices))
  val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_val, sampler=SubsetRandomSampler(val_indices))
  test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_val, shuffle=False)

## Calculate flatten size of each input data
# args.input_size = np.prod(train_dataset[0][0].size())
# args.input_size = train_dataset[0][0].size()[0]
# print(args.input_size)
#########################################################
## Train and Test Model
#########################################################
gmvae = GMVAE(args)

## Pre-training Phase
# gmvae.pre_train(train_loader)

## Training Phase
history_loss = gmvae.train(train_loader, val_loader, test_loader)

# Iterative Training Phase
# gmvae.iterative_train(train_loader, val_loader, test_loader, epochs = 79)

# Testing Phase
precision, recall, f1_score, ari = gmvae.test(test_loader, output_result = True)

print("Valid - Precision: {:.5f}; Recall: {:.5f}; F1_score: {:.5f}; ARI: {:.5f}".format(precision, recall, f1_score, ari))

# Kmeans fit Phase
# precision, recall, f1_score, ari = gmvae.k_means(test_loader)

# print("Kmeans - Precision: {:.5f}; Recall: {:.5f}; F1_score: {:.5f}; ARI: {:.5f}".format(precision, recall, f1_score, ari))

# GMM fit Phase
# precision, recall, f1_score, ari = gmvae.fit_gmm(test_loader)

# print("GMM - Precision: {:.5f}; Recall: {:.5f}; F1_score: {:.5f}; ARI: {:.5f}".format(precision, recall, f1_score, ari))

# Generate the latents
# latents, contignames = gmvae.generate_latent(test_loader)
# np.save('data/latents', latents)
# with open("data/contignames.pkl", "wb") as f:
#   pickle.dump(contignames, f)

# plot latent space
# gmvae.plot_latent_space(test_loader, True)

# print("Testing phase...")
# print("Accuracy: %.5lf, NMI: %.5lf" % (accuracy, nmi) )

