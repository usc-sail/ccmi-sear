from time import time
import logging
import os
import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, exp_dir, n_iter, batch_size):
        self.batch_time = AverageMeter()
        self.per_sample_time = AverageMeter()
        self.loss_meter = None
        self.best_epoch = 0
        self.best_loss = -np.inf
        self.epoch_start_time = time()
        self.batch_start_time = time()
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.exp_dir = exp_dir
        logfile = os.path.join(exp_dir, 'log.txt')
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(filename=logfile, filemode='a', 
                                level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger()

    def update(self, loss):
        if self.loss_meter is None:
            self.loss_meter = {k: AverageMeter() for k in loss.keys()}
        
        update_time = time() - self.batch_start_time
        for k, loss_ in loss.items():
            self.loss_meter[k].update(loss_.item(), self.batch_size)
        self.batch_time.update(update_time)
        self.per_sample_time.update(update_time/self.batch_size)
        self.batch_start_time = time()
        
    def print_stats(self, epoch, itr):
        logstr = f"Epoch: [{epoch}][{itr}/{self.n_iter}]\tBatch Time {self.batch_time.avg:.2f}\tPer sample Time {self.per_sample_time.avg:.4f}\t"
        for k, v in self.loss_meter.items():
            logstr += f'{k}: {v.avg:.3f}\t'
                    
        self.logger.info(logstr)  
    
    def save_model(self, model, itr, mode):
        save_path = os.path.join(self.exp_dir, 'models', f'model.{mode}.{itr}.pth')
        torch.save(model.state_dict(), save_path)

    def print_validation_stats(self, model, val_loss, epoch, global_step):
        if val_loss['loss'] < self.best_loss:
            self.best_loss = val_loss['loss']
            self.best_epoch = epoch
            self.save_model(model, itr=global_step, mode='iter')
        self.logger.info('-'*15 + f' step {global_step} validation ' + '-'*15)
        logstr = 'Training stats:\t'
        for k, v in self.loss_meter.items():
            logstr += f'{k}: {v.avg:.3f}\t'
        self.logger.info(logstr)
        logstr = 'Validation stats:\t'
        for k, v in val_loss.items():
            logstr += f'{k}: {v}\t' 
        self.logger.info(logstr)

    def epoch_end(self, model, epoch):
        epoch_time = time() - self.epoch_start_time
        self.logger.info(f'Epoch {epoch} training time: {epoch_time:.2f}')
        self.save_model(model, itr=epoch, mode='epoch')
        self.batch_time.reset()
        self.per_sample_time.reset()
        for k in self.loss_meter.keys():
            self.loss_meter[k].reset()
        self.epoch_start_time = time()    
