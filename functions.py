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
from pytorch_models import ConvBatchLeaky1D,ConvGLU
from scipy.signal import spectrogram
import pandas as pd

''' data generator '''

# 2.0801182, 1.6936827 come from sample mean, std

normalize = lambda x: np.tanh(x/8-0.6)
normalize_log = lambda x: np.tanh((np.log(x+1e-8)-2.0801182)/1.6936827)

class SequenceGenSpec(Dataset):
    tot = 629145480
    def __init__(self,data,IsTrain=True,train_cutoff=500000000,length=150000,Is2D=True):
        # data is numpy array of (value, time)
        # data[0:cutoff] used for train, the rest used for validation
        self.data = data
        self.IsTrain = IsTrain
        self.train_cutoff = train_cutoff
        self.length = length
        self.index_max = train_cutoff - length - 1
        self.Is2D = Is2D
        
    def __len__(self):
        return int(self.train_cutoff/self.length) if self.IsTrain else int((self.tot - self.train_cutoff)/self.length)-1

    def __getitem__(self, idx):
        r = np.random.randint(0,self.index_max) if self.IsTrain else self.train_cutoff + self.length*idx
        x = self.data[r:r+self.length,0]
        _,_,x = spectrogram(x,nperseg=256,noverlap=256//4)
        x = normalize_log(x)
        y = self.data[r+self.length,1]
        return (x if self.Is2D else x[np.newaxis]),y

class SequenceGenSpecTest(Dataset):
    def __init__(self,seg_id,Is2D=True):
        # seg_id is a list of seg id from submission file
        self.seg_id = seg_id
        self.Is2D = Is2D
        
    def __len__(self):
        return len(self.seg_id)

    def __getitem__(self, idx):
        x = pd.read_csv('../Data/test/'+self.seg_id[idx]+'.csv',dtype={'acoustic_data': np.float32}).values.flatten()
        _,_,x = spectrogram(x,nperseg=256,noverlap=256//4)
        x = normalize_log(x)
        return (x if self.Is2D else x[np.newaxis])

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

class SequenceGenLM(Dataset):
    # pretain as language model style. Used as base for fine tune
    tot = 629145480
    def __init__(self,data,IsTrain=True,train_cutoff=500000000,length=150000,predict_len=200):
        # data is numpy array of (value, time)
        # data[0:cutoff] used for train, the rest used for validation
        self.data = data
        self.IsTrain = IsTrain
        self.train_cutoff = train_cutoff
        self.length = length
        self.predict_len = predict_len
        self.tot_len = length+predict_len
        self.index_max = train_cutoff - self.tot_len -1
        
    def __len__(self):
        return int(self.train_cutoff/self.tot_len) if self.IsTrain else int((self.tot - self.train_cutoff)/self.tot_len)-1

    def __getitem__(self, idx):
        r = np.random.randint(0,self.index_max) if self.IsTrain else (self.train_cutoff + self.tot_len*idx)
        x = self.data[r:r+self.length,0]
        y = self.data[r+self.length:r+self.tot_len,0]
        return x[np.newaxis],y

class SequenceGenTest(Dataset):
    def __init__(self,seg_id):
        # seg_id is a list of seg id from submission file
        self.seg_id = seg_id
        
    def __len__(self):
        return len(self.seg_id)
    
    def __getitem__(self, idx):
        x = pd.read_csv('../Data/test/'+self.seg_id[idx]+'.csv',dtype={'acoustic_data': np.float32}).values.flatten()
        x = normalize(x)
        return x[np.newaxis]
    
class SequenceGenNojump(Dataset):
    # dis-allow one observation to go over an quake
    tot = 629145480
    tot_jump = 17
    def __init__(self,data,jump_index,IsTrain=True,trainJump=13,length=150000):
        # data is numpy array of (value, time)
        # data[0:cutoff] used for train, the rest used for validation
        self.data = data
        self.IsTrain = IsTrain
        self.trainJump = trainJump
        self.length = length
        self.jump_index = jump_index
        
    def __len__(self):
        return int(self.jump_index[self.trainJump]/self.length) if self.IsTrain else int((self.tot - self.jump_index[self.trainJump])/self.length)

    def __getitem__(self, idx):
        idx = np.random.randint(0,self.trainJump) if self.IsTrain else np.random.randint(self.trainJump,self.tot_jump)
        r = np.random.randint(self.jump_index[idx]+2,self.jump_index[idx+1]-self.length-2)
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
    
class DenseBlock2(nn.Module):
    # N,D,L to N,2*D,L
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = ConvBatchLeaky1D(in_channel,in_channel,3,padding=1)
        self.conv2 = ConvBatchLeaky1D(in_channel*2,in_channel*2,3,stride=2)
        
    def forward(self, x):
        return self.conv2(torch.cat([x,self.conv1(x)],1))
    
class DenseBlockGLU(nn.Module):
    # N,D,L to N,2*D,L/2
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = ConvGLU(in_channel,in_channel,3,padding=1)
        self.conv2 = ConvBatchLeaky1D(in_channel*2,in_channel*2,3,stride=2)
        
    def forward(self, x):
        return self.conv2(torch.cat([x,self.conv1(x)],1))

class ResidualBlockGLU(nn.Module):
    # N,D,L to N,2*D,L/2
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = ConvGLU(in_channel,in_channel,3,padding=1)
        self.conv2 = ConvBatchLeaky1D(in_channel,in_channel*2,3,stride=2)
        
    def forward(self, x):
        return self.conv2(x+self.conv1(x))    
    
class CNN_RNN2seq(nn.Module):
    def __init__(self,conv,linear):
        super().__init__()
        self.conv = conv 
        self.linear = linear
        
    def forward(self,x):
        n = x.shape[0]
        _,x = self.conv(x)
        x = x.contiguous().view(n,-1)
        x = self.linear(x)
        return x.squeeze(1)    

def loss_func_generator(distanceFun):
    def loss_func(model,data):
        X,y = data
        yhat = model(X)
        loss = distanceFun(yhat,y)
        return loss
    return loss_func