
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow.keras.backend as K
# from skimage.segmentation import mark_boundaries
import os, glob, sys, importlib
from functools import partial
from tqdm import tqdm
from datetime import datetime

import torch.nn as nn
import torch
from torch.autograd import Variable
from torchsummary import summary
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from .u2net_loss import muti_bce_loss_fusion
import torch.nn as nn

from . import U2Net
# import U2Net

class Trainer():
    def __init__(self, dataloader, checkpoint_manager, train_length = 100, 
                 device = "cuda", model_args = {}, tb_writer = None,
                 metrics = {"loss": np.inf}, augmenter = None,
                clip_grad = False):
        self.checkpoint_manager = checkpoint_manager
        self.train_length = train_length
        self.loss = muti_bce_loss_fusion
        self.dataloader = dataloader
        self.epoch = 0
        self.train_step_index = 0
        self.clip_grad = clip_grad
        self.model_args = model_args.copy()
        self.device = device
        self.tb_writer = tb_writer
        self.metrics = metrics.copy()
        if augmenter is None:
            self.augmenter = lambda x: x
        else:
            self.augmenter = augmenter
        
        self.__build__()
        self.load_status()
        
    def __build__(self):

        model = U2Net.U2NET_full(**self.model_args)
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.99)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 200, gamma=0.99)
        
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def save_status(self):
        checkpoint_dict = self.checkpoint_manager.get_save_paths_new(index = self.train_step_index, metrics = self.metrics)
        save_dict = {
                    'train_epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'lr_state_dict': self.lr_scheduler.state_dict(),
                    'train_step_index': self.train_step_index,
                    }
        save_dict.update(self.metrics)
        for checkpoint_path in checkpoint_dict.values():
            torch.save(save_dict, checkpoint_path)

    def load_status(self):
        checkpoint_path = self.checkpoint_manager.get_save_paths()["last_period"]
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            return
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['train_epoch']
        self.train_step_index = checkpoint['train_step_index']
        # val_step = checkpoint['val_step']
        
        
    def train_step(self, data, train_step_index):
        self.model.train()
        # inputs, labels, usable = data
        inputs, labels = data
        # if any(usable < 0.):
        #     continue

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if self.device == "cuda":
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        self.optimizer.zero_grad()

        d0, d1, d2, d3, d4, d5, d6 = self.model(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 'inf', norm_type = 1.)
        self.optimizer.step()
        self.lr_scheduler.step()

        # # print statistics
        loss_value = loss.data.item()
        loss_tar =  loss2.data.item()
        # running_loss += loss_value
        # running_tar_loss += loss_tar
        if self.tb_writer is not None:
            self.tb_writer.add_scalars('Training Loss',
            {'Training loss' : loss_value },
            self.train_step_index + 1)

        self.metrics["loss"] = loss_value
        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            
    def train(self):
        
        for idx, data in enumerate(self.dataloader):
            self.train_step(data, self.train_step_index)
            self.train_step_index += 1
            
            if idx >= self.train_length - 1:
                break
        self.epoch += 1
        self.save_status()

    def execute(self):
        self.train()
        
    