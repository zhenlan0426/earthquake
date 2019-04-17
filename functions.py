#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 16:03:44 2019

@author: will
"""
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.nn import Linear
from torch import nn
import torch
from pytorch_models import ConvBatchLeaky1D,ConvGLU,ConvBatchLeaky,Conv2dGLU,ConvBatchRRelu1D
from pytorch_util import predict
from scipy.signal import spectrogram
import pandas as pd

''' data generator '''

# 2.0801182, 1.6936827 come from sample mean, std

normalize = lambda x: np.tanh(x/8-0.6)
normalize_log = lambda x: np.tanh((np.log(x+1e-8)-2.0801182)/1.6936827)
normalize2 = lambda x: (x - 4.876247882843018)/6.380820274353027
normalize_log2 = lambda x: (np.log(x+1e-8)-2.0801182)/1.6936827

class weightedAverageGen(Dataset):
    def __init__(self,filePath,y_path=None):
        # seg_id is a list of seg id from submission file
        self.xs = np.stack([np.load(file) for file in filePath],1)
        if y_path is not None:
            self.y = np.load(y_path)
        else:
            self.y = None
        
    def __len__(self):
        return self.xs.shape[0]
    
    def __getitem__(self, idx):
        return self.xs[idx] if self.y is None else (self.xs[idx], self.y[idx])

class SequenceGenSpec(Dataset):
    tot = 629145480
    def __init__(self,data,IsTrain=True,train_cutoff=500000000,length=150000,Is2D=True,normalFun=normalize_log):
        # data is numpy array of (value, time)
        # data[0:cutoff] used for train, the rest used for validation
        self.data = data
        self.IsTrain = IsTrain
        self.train_cutoff = train_cutoff
        self.length = length
        self.index_max = train_cutoff - length - 1
        self.Is2D = Is2D
        self.normalFun = normalFun        
        
    def __len__(self):
        return int(self.train_cutoff/self.length) if self.IsTrain else int((self.tot - self.train_cutoff)/self.length)-1

    def __getitem__(self, idx):
        r = np.random.randint(0,self.index_max) if self.IsTrain else self.train_cutoff + self.length*idx
        x = self.data[r:r+self.length,0]
        _,_,x = spectrogram(x,nperseg=256,noverlap=256//4)
        x = self.normalFun(x)
        y = self.data[r+self.length,1]
        return (x if self.Is2D else x[np.newaxis]),y

class SequenceGenSpecTest(Dataset):
    def __init__(self,seg_id,normalFun,Is2D=True):
        # seg_id is a list of seg id from submission file
        self.seg_id = seg_id
        self.Is2D = Is2D
        self.normalFun = normalFun
        
    def __len__(self):
        return len(self.seg_id)

    def __getitem__(self, idx):
        x = pd.read_csv('../Data/test/'+self.seg_id[idx]+'.csv',dtype={'acoustic_data': np.float32}).values.flatten()
        _,_,x = spectrogram(x,nperseg=256,noverlap=256//4)
        x = self.normalFun(x)
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

class SequenceGenSample(Dataset):
    tot = 629145480
    def __init__(self,data,IsTrain=True,train_cutoff=500000000,length=150000,sample_intval=4):
        # data is numpy array of (value, time)
        # data[0:cutoff] used for train, the rest used for validation
        self.data = data
        self.IsTrain = IsTrain
        self.train_cutoff = train_cutoff
        self.length = length
        self.sample_intval = sample_intval
        self.index_max = train_cutoff - length - 1
        
    def __len__(self):
        return int(self.train_cutoff/self.length) if self.IsTrain else (int((self.tot - self.train_cutoff)/self.length)-1)*self.sample_intval

    def __getitem__(self, idx):
        if self.IsTrain:
            r = np.random.randint(0,self.index_max)
        else:
            i = idx//self.sample_intval
            j = idx%self.sample_intval
            r = self.train_cutoff + self.length*i + j
        x = self.data[r:r+self.length:self.sample_intval,0]
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
    def __init__(self,seg_id,normalFun):
        # seg_id is a list of seg id from submission file
        self.seg_id = seg_id
        self.normalFun = normalFun
        
    def __len__(self):
        return len(self.seg_id)
    
    def __getitem__(self, idx):
        x = pd.read_csv('../Data/test/'+self.seg_id[idx]+'.csv',dtype={'acoustic_data': np.float32}).values.flatten()
        x = self.normalFun(x)
        return x[np.newaxis]
 
class SequenceGenTestSample(Dataset):
    def __init__(self,seg_id,normalFun,sample_intval=4):
        # seg_id is a list of seg id from submission file
        self.seg_id = seg_id
        self.sample_intval = sample_intval
        self.normalFun = normalFun
        
    def __len__(self):
        return len(self.seg_id) * self.sample_intval
    
    def __getitem__(self, idx):
        x = pd.read_csv('../Data/test/'+self.seg_id[idx//self.sample_intval]+'.csv',dtype={'acoustic_data': np.float32}).values.flatten()
        x = self.normalFun(x[idx%self.sample_intval::self.sample_intval])
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


class SequenceGenBoostSample(Dataset):
    def __init__(self,data,start,iteration,length=150000,sample_intval=4,returnY=False):
        self.data = data
        self.start = start
        self.iteration = iteration
        self.length = length
        self.sample_intval = sample_intval
        self.returnY = returnY
        
    def __len__(self):
        return self.iteration * self.sample_intval

    def __getitem__(self, idx):
        i = idx//self.sample_intval
        j = idx%self.sample_intval
        r = self.start + self.length*i + j
        x = self.data[r:r+self.length:self.sample_intval,0]
        return (x[np.newaxis],self.data[r+self.length,1]) if self.returnY else x[np.newaxis]


class SequenceGenSpecBoost(Dataset):
    def __init__(self,data,start,iteration,length=150000,Is2D=True,normalFun=normalize_log,returnY=False):
        # data is numpy array of (value, time)
        # data[0:cutoff] used for train, the rest used for validation
        self.data = data
        self.start = start
        self.iteration = iteration
        self.length = length
        self.Is2D = Is2D
        self.normalFun = normalFun        
        self.returnY = returnY
        
    def __len__(self):
        return self.iteration

    def __getitem__(self, idx):
        r = self.start + self.length*idx
        x = self.data[r:r+self.length,0]
        _,_,x = spectrogram(x,nperseg=256,noverlap=256//4)
        x = self.normalFun(x)
        x = x if self.Is2D else x[np.newaxis]
        y = self.data[r+self.length,1]
        return (x,y) if self.returnY else x

''' build models '''    

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

class ResidualBlockRRelu(nn.Module):
    # N,D,L to N,2*D,L
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = ConvBatchRRelu1D(in_channel,in_channel,3,padding=1)
        self.conv2 = ConvBatchRRelu1D(in_channel,in_channel*2,3,stride=2)
        
    def forward(self, x):
        return self.conv2(x+self.conv1(x))

class ResidualBlock(nn.Module):
    # N,D,L to N,2*D,L/2
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = ConvBatchLeaky1D(in_channel,in_channel,3,padding=1)
        self.conv2 = ConvBatchLeaky1D(in_channel,in_channel*2,3,stride=2)
        
    def forward(self, x):
        return self.conv2(x+self.conv1(x))    

class ResidualBlock3d(nn.Module):
    # N,C,F,L to N,2*C,F/2,L/2
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = ConvBatchLeaky(in_channel,in_channel,3,padding=1)
        self.conv2 = ConvBatchLeaky(in_channel,in_channel*2,3,stride=2)
        
    def forward(self, x):
        return self.conv2(x+self.conv1(x))  

class DenseBlock3d(nn.Module):
    # N,C,F,L to N,2*C,F/2,L/2
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = ConvBatchLeaky(in_channel,in_channel,3,padding=1)
        self.conv2 = ConvBatchLeaky(in_channel*2,in_channel*2,3,stride=2)
        
    def forward(self, x):
        return self.conv2(torch.cat([x,self.conv1(x)],1))  

class ResidualBlockGLU3d(nn.Module):
    # N,C,F,L to N,2*C,F/2,L/2
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = Conv2dGLU(in_channel,in_channel,3,padding=1)
        self.conv2 = ConvBatchLeaky(in_channel,in_channel*2,3,stride=2)
        
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

class CNN_RNN2seq_transpose(nn.Module):
    def __init__(self,conv,linear):
        super().__init__()
        self.conv = conv 
        self.linear = linear
        self.convert = Linear(1,1)
        
    def forward(self,x):
        n = x.shape[0]
        _,x = self.conv(x)
        x = x.contiguous().view(n*2,-1)
        x = self.linear(x).squeeze(1)
        x = x.view(n,2).mean(1,keepdim=True)
        return self.convert(x).squeeze(1)

def loss_func_generator(distanceFun):
    def loss_func(model,data):
        X,y = data
        yhat = model(X)
        loss = distanceFun(yhat,y)
        return loss
    return loss_func


def make_submission_sample(name,model,normalFun,batch_size):
    submission = pd.read_csv('../Data/sample_submission.csv')
    test_gen = SequenceGenTestSample(submission.seg_id.tolist(),normalFun)
    test_gen = DataLoader(test_gen,batch_size,False,num_workers=2)
    yhat = np.maximum(predict(model,test_gen),0)
    submission.iloc[:,1] = np.median(yhat.reshape(-1,4),1)
    submission.to_csv('../Submission/'+name+'.csv',index=False)
    
def make_submission(name,model,normalFun,batch_size):
    submission = pd.read_csv('../Data/sample_submission.csv')
    test_gen = SequenceGenSpecTest(submission.seg_id.tolist(),normalFun)
    test_gen = DataLoader(test_gen,batch_size,False,num_workers=2)
    submission.iloc[:,1] = np.maximum(predict(model,test_gen),0)
    submission.to_csv('../Submission/'+name+'.csv',index=False)
     
def save_ModelandValidation_spec(model,train,name,normalFun,batch_size=16):
    train_gen = SequenceGenSpecBoost(train,0,int(500000000/150000),normalFun=normalFun)
    train_gen = DataLoader(train_gen,batch_size,False)
    val_gen = SequenceGenSpecBoost(train,500000000,int((train.shape[0] - 500000000)/150000)-1,normalFun=normalFun)
    val_gen = DataLoader(val_gen,batch_size,False)
    ytrain = predict(model,train_gen)
    yval = predict(model,val_gen)
    np.save('../Model/'+name+'_train.npy', ytrain)
    np.save('../Model/'+name+'_val.npy', yval)     
    torch.save(model.state_dict(), '../Model/'+name+'.pt')  
        
def save_ModelandValidation_Raw(model,train,name,batch_size=16):
    train_gen = SequenceGenBoostSample(train,0,int(500000000/150000))
    train_gen = DataLoader(train_gen,batch_size,False)
    val_gen = SequenceGenBoostSample(train,500000000,int((train.shape[0] - 500000000)/150000)-1)
    val_gen = DataLoader(val_gen,batch_size,False)
    ytrain = predict(model,train_gen)
    yval = predict(model,val_gen)
    np.save('../Model/'+name+'_train.npy', np.median(ytrain.reshape(-1,4),1))
    np.save('../Model/'+name+'_val.npy', np.median(yval.reshape(-1,4),1)) 
    torch.save(model.state_dict(), '../Model/'+name+'.pt')    
    
    

    
    
    
    
    
    
    
