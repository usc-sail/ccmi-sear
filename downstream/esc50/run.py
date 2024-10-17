# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import torch
import random
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader
from models import FCModel
import numpy as np
from traintest import train, validate

def make_deterministic(seed=1994):
    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--meta-train", type=str, default=None, help="training data json")
parser.add_argument("--meta-val", type=str, default=None, help="validation data json")
parser.add_argument("--meta-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="testing data json")
parser.add_argument("--label-csv", type=str, default=None, help="csv metadata")
parser.add_argument("--embedding-dim", type=int, default=512, help="hidden dim for lstm")
parser.add_argument("--n-layers", type=int, default=1, help="num. lstm layers")
parser.add_argument("--n_class", type=int, default=527, help="number of classes")

parser.add_argument('--inp-dim', type=int)
parser.add_argument("--num-training-iters", type=int, default=1)
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
# only used in pretraining stage or from-scratch fine-tuning experiments

parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the models or not', type=ast.literal_eval)

# fine-tuning arguments
parser.add_argument("--pretrained_mdl_path", type=str, default=None, help="the ssl pretrained models path")
parser.add_argument("--metrics", type=str, default="acc", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])

args = parser.parse_args()
make_deterministic()

train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.meta_train, args.data_train, label_csv=args.label_csv),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, drop_last=True)

val_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.meta_val, args.data_val, label_csv=args.label_csv), 
        batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=False)


audio_model = FCModel(args.inp_dim, args.n_class, args.embedding_dim)
if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = torch.nn.DataParallel(audio_model)

train(audio_model, train_loader, val_loader, args)

