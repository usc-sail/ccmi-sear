# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import time
import torch
import dataloader
from models import blip_pretrain
import numpy as np
import random
from trainer import train
from datetime import datetime
from omegaconf import OmegaConf
from logger import Logger
import sys

def make_deterministic(seed=123):
    # Seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default=None, help="training data json")
parser.add_argument("--data-val", type=str, default=None, help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--config-file", type=str, default=None, help="config file")
args = parser.parse_args()


config = OmegaConf.load(args.config_file)
data_cfg = config.data
model_cfg = config.model
run_cfg = config.run

make_deterministic(seed=run_cfg.seed)

now = datetime.now()
exp_dir = run_cfg.get('output_dir', 'exp/' + now.strftime("%d.%m.%Y__%H.%M"))
#print(f"\nCreating experiment directory: {exp_dir}")
assert os.path.exists(f"{exp_dir}/models") == False, "Experiment directory exists, exiting"
os.makedirs(f"{exp_dir}/models")
with open(os.path.join(exp_dir, 'config.yaml'), 'w') as fp:
    OmegaConf.save(config, fp)


train_loader = torch.utils.data.DataLoader(
        dataloader.AudioDataset(args.data_train, data_conf=data_cfg),
        batch_size=run_cfg.batch_size, shuffle=True, num_workers=run_cfg.num_workers, pin_memory=False, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.AudioDataset(args.data_val, data_conf=data_cfg),
    batch_size=run_cfg.batch_size*4, shuffle=False, num_workers=run_cfg.num_workers, pin_memory=False, drop_last=True)

logger = Logger(exp_dir=run_cfg.output_dir, n_iter=len(train_loader), batch_size=run_cfg.batch_size)
sys.stdout.write = logger.logger.info
print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))
print('Now train with {:d} training samples, evaluate with {:d} samples'.format(len(train_loader.dataset), len(val_loader.dataset)))

model = blip_pretrain.BlipPretrain.from_config(model_cfg)

train(model, config, logger, train_loader, val_loader)
