#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:03:44 2019

@author: will
"""
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import torch
from pytorch_models import ConvBatchLeaky1D

''' data generator '''

normalize = lambda x: np.tanh(x/8-0.6)

class SequenceGen(Dataset):
    tot = 629145480
    def __init__(self,data,IsTrain=True,train_cutoff=500000000,length=150000):
        # data is numpy array of (value, time)
        # data[0:cutoff] used for train, the rest used for validation
        self.data = data
        self.IsTrain = IsTrain
        self.train_cutoff = train_cutoff
        self.length = length
        self.index_max = train_cutoff - length - 1
        
    def __len__(self):
        return int(self.train_cutoff/self.length) if self.IsTrain else int((self.tot - self.train_cutoff)/self.length)-1

    def __getitem__(self, idx):
        r = np.random.randint(0,self.index_max) if self.IsTrain else self.train_cutoff + self.length*idx
        x = self.data[r:r+self.length,0]
        y = self.data[r+self.length,1]
        return x[np.newaxis],y
    
    
class DenseBlock(nn.Module):
    # N,D,L to N,2*D,L
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = ConvBatchLeaky1D(in_channel,in_channel*2,1)
        self.conv2 = ConvBatchLeaky1D(in_channel*2,in_channel,3,padding=1)
        self.maxpool = nn.MaxPool1d(3,2)
        
    def forward(self, x):
        return self.maxpool(torch.cat([x,self.conv2(self.conv1(x))],1))
    
    
class CNN_RNN2seq(nn.Module):
    def __init__(self,conv,linear):
        super().__init__()
        self.conv = conv 
        self.linear = linear
        
    def forward(self,x):
        n = x.shape[0]
        _,x = self.conv(x)
        x = x.contiguous().view(n,2*512)
        x = self.linear(x)
        return x.squeeze(1)    

def loss_func_generator(distanceFun):
    def loss_func(model,data):
        X,y = data
        yhat = model(X)
        loss = distanceFun(yhat,y)
        return loss
    return loss_func