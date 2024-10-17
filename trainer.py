# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from utils.optims import *
from utils.registry import registry

def init_optimizer(model, run_cfg, logger):
    weight_decay = run_cfg.weight_decay
    lr_scale = run_cfg.lr_layer_decay
    optim_params = model.get_optimizer_params(weight_decay, lr_scale)
    num_params = 0

    for p_group in optim_params:
        for p in p_group['params']:
            num_params += p.data.nelement()
    print('number of trainable parameters: {} million'.format(num_params/1e6))
    beta2 = run_cfg.beta2
    opt = torch.optim.AdamW(
                    optim_params,
                    lr=float(run_cfg.init_lr),
                    betas=(0.9, beta2),
                  )

    return opt


def train(model, config, logger, train_loader, val_loader, test_loader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)
    run_cfg = config.run

    global_step = 0
    max_epoch = run_cfg.max_epoch
    batch_size = run_cfg.batch_size
    n_iter = len(train_loader)
    
    scaler = torch.cuda.amp.GradScaler() if run_cfg.amp else None
    use_amp = scaler is not None
    accum_grad_iters = run_cfg.accum_grad_iters    

    optimizer = init_optimizer(model, run_cfg, logger)

    scheduler_cls = registry.get_lr_scheduler_class(run_cfg.lr_sched)
    scheduler = scheduler_cls(
                    optimizer=optimizer,
                    max_epoch=run_cfg.max_epoch,
                    min_lr=run_cfg.min_lr,
                    init_lr=run_cfg.init_lr,
                    decay_rate=run_cfg.lr_decay_rate,
                    warmup_start_lr=run_cfg.warmup_lr,
                    warmup_steps=run_cfg.warmup_steps
                    )

    if run_cfg.save_iters > 0:
        save_iter = run_cfg.save_iters
    else:
        save_iter = len(train_loader)
    
    model = torch.nn.DataParallel(model)
    model.to(device)
    print("start training...")

    for epoch in range(max_epoch):
        begin_time = time.time()
        end_time = time.time()
        model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, caption) in enumerate(train_loader):
            audio_input = audio_input.to(device, non_blocking=True) 
            
            samples = {
                        'spectrogram': audio_input,
                        'caption': caption,
                        'epoch': epoch,
                        'num_iters_per_epoch': n_iter,
                        'iters': i
                      }
            
            scheduler.step(cur_epoch=epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(samples)
                loss_dict = {k: v for k,v in output.items() if 'loss' in k}
                loss = output['loss']
                loss /= accum_grad_iters
            
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
            
            # record loss
            logger.update(loss_dict)

            print_step = global_step % run_cfg.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (run_cfg.n_print_steps/10) == 0
            print_step = (print_step or early_print_step) and global_step > 0

            if print_step:
                logger.print_stats(epoch, i)
           
            global_step += 1
        
            if global_step % save_iter == 0:
                valid_loss = evaluate(model, val_loader)
                
                logger.print_validation_stats(model, valid_loss, epoch, global_step)
            
        logger.epoch_end(model, epoch)
    
    if test_loader is not None:
        pass

    


def evaluate(model, eval_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    loss = {}
    with torch.no_grad():
        for i, (audio_input, caption) in enumerate(eval_loader):
            audio_input = audio_input.to(device)
            
            samples = {
                        'spectrogram': audio_input,
                        'caption': caption,
                        'epoch': 10,
                        'num_iters_per_epoch': 100,
                        'iters': 0
                      }
            # compute output
            output = model(samples)
            loss_dict = {k: v.item() for k,v in output.items() if 'loss' in k}
            
            for k, v in loss_dict.items():
                if k not in loss:
                    loss[k] = [v]
                else:
                    loss[k].append(v)

        loss = {k: np.mean(v) for k, v in loss.items()}
    return loss
