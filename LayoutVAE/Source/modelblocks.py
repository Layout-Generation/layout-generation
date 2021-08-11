# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:14:02 2021

@author: Tushar & Tanishk
"""
from __future__ import division
import torch as T
from torch.nn import  Sequential , Linear , ReLU , PoissonNLLLoss, LSTM
from torch.distributions import MultivariateNormal 

class fcblock(T.nn.Module):
    def __init__(self, n_class):
        super(fcblock, self).__init__()
        self.seq = Sequential(
            Linear(n_class,128),
            ReLU(),
            Linear(128,128),
            ReLU(),
        )
    def forward(self,inputs):
        out = self.seq(inputs)
        return out

class Embeder(T.nn.Module):
    def __init__(self,n_class):
        super(Embeder,self).__init__()
        
        self.fcb1 = fcblock(n_class)
        self.fcb2 = fcblock(n_class)
        self.fcb3 = fcblock(n_class)
        self.fc   = Linear(128*3,128)

 
    def forward(self,inputs):
        in1,in2,in3 = inputs
        in1 = self.fcb1(in1)
        in2 = self.fcb2(in2)
        in3 = self.fcb3(in3)
        out = T.cat((in1,in2,in3),1)
        out = self.fc(out)
        return out

class Encoder(T.nn.Module):
    def __init__(self, in_dim=1 ,latent_dim=32):
        super(Encoder,self).__init__()
        self.act = ReLU()
        self.fc1 = Linear(in_dim,128)
        self.fc2 = Linear(128,128)
        self.fc3 = Linear(256,latent_dim)
        self.fc4 = Linear(latent_dim,latent_dim)
        self.fc5 = Linear(latent_dim,latent_dim)
        
    def forward(self,inputs):
        in1,in2 = inputs
        out = self.fc1(in1)
        out = self.act(out)
        out = self.fc2(out)
        out = T.cat((out,in2),1)
        out = self.fc3(out)
        out = self.act(out)
        mu  = self.fc4(out)
        logvar = self.fc5(out)
        return mu,logvar


class Prior(T.nn.Module):
    def __init__(self,latent_dim=32):
        super(Prior,self).__init__()
        
        self.act = ReLU()
        self.fc1 = Linear(128,latent_dim)
        self.fc2 = Linear(latent_dim,latent_dim)
        self.fc3 = Linear(latent_dim,latent_dim)
        
    def forward(self,inputs):
        out = inputs
        out = self.fc1(out)
        out = self.act(out)
        mu  = self.fc2(out)
        logvar = self.fc3(out)  
        return mu,logvar

class Decoder(T.nn.Module):
    def __init__(self,output_dim,latent_dim=32):
        super(Decoder,self).__init__()
        self.act = ReLU()
        self.fc1 = Linear(128+latent_dim,128)
        self.fc2 = Linear(128,64)
        self.fc3 = Linear(64,output_dim)
        
    def forward(self,inputs):
        in1,in2 = inputs
        out = T.cat((in1,in2),1)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        return out

"""# LOSS FUNCTION"""

class ELBOLoss(T.nn.Module):

    def __init__(self):
        super(ELBOLoss,self).__init__()
    
    def forward(self,inputs):
        mu1, logvar1, mu2, logvar2 , in1, in2 = inputs

        mask = (in2>0)+0.0
        in2 = in2-mask

        '''KL Divergence'''
        kl =   0.5 * T.sum((logvar2 - logvar1) - 1 + (logvar1.exp() + (mu2 - mu1).pow(2) )/logvar2.exp() , dim = 1).mean()
        
        '''Poisson Negative Log Likelihood'''
        pnll = PoissonNLLLoss()(in1,in2)

        loss = kl+pnll
        
        return loss, pnll , kl
 


class EmbedBbox(T.nn.Module):
    
    def __init__(self,n_class):
        super(EmbedBbox,self).__init__()
       
        self.fcb1 = fcblock(n_class)
        self.fcb2 = fcblock(n_class)
        self.seq1 = Sequential(
            Linear(128,128),
            ReLU()
        )
        
        self.n_class = n_class
        self.fc   = Linear(128*3,128)
        self.lstm = LSTM(n_class+4, hidden_size=128)

    def forward(self,inputs):
        
        in1,in2,in3 = inputs

        _ , (h_0 , c_0 ) = self.lstm(in3)
        hn  = h_0.view(-1, 128)
        
        in1 = self.fcb1(in1)
        in2 = self.fcb2(in2)
        in3 = self.seq1(hn)
        
        out = T.cat((in1,in2,in3),1)
        out = self.fc(out)
        
        return out
    
class ELBOLoss_Bbox(T.nn.Module):
    
    def __init__(self):
        super(ELBOLoss_Bbox,self).__init__()
    
    def forward(self,inputs):
        mu1,logvar1,mu2,logvar2, xp , yp = inputs
        
        ''' KL Divergence '''
        kl =   0.5 * T.sum((logvar2 - logvar1) - 1 + (logvar1.exp() + (mu2 - mu1).pow(2) )/logvar2.exp() , dim = -1 ).mean()
        
        ''' Multivariate Guassian Likelihood '''
        mse = T.nn.MSELoss()(xp,yp)
        loss = mse + kl
        
        return loss, kl,mse


class Reparamatrize_bvae(T.nn.Module):
    
    def __init__(self):
        super(Reparamatrize_bvae,self).__init__()
    
    def forward(self,inputs):
        
        mu , logvar = inputs
        std = T.exp(logvar/2)
        eps = T.rand_like(std)

        return eps*std + mu
        

class ReparamatrizeMulti(T.nn.Module):
    
    def __init__(self):
        super(ReparamatrizeMulti,self).__init__()
    
    def forward(self,inputs):
       
        mu  = inputs
        std = (T.ones_like(mu)*0.02)
        eps = T.rand_like(std)
        
        return eps*std + mu
    
class Reparamatrize_cvae(T.nn.Module):
    
    def __init__(self):
        super(Reparamatrize_cvae,self).__init__()
        
    def forward(self,inputs):
        
        mu , logvar = inputs
        '''
        mu = mean 
        logvar = log of diagonal elements of covariance matrix
        '''
        # Covarince Matrix
        covar  = T.diag_embed(T.exp(logvar/2), dim1=-2,dim2=-1)

        # Multivariate Normal Distribution
        p = MultivariateNormal(mu,covar)
        z_latent = p.rsample().float()
        return z_latent

class Sampling(T.nn.Module):

    def __init__(self,MAX_BOX):
        super(Sampling,self).__init__()
        self.max_box = MAX_BOX
    
    def forward(self,lamda):
        
        lamda   = lamda.view(-1)
        mask    = T.zeros(lamda.shape[0] , self.max_box)
        lamda   = T.t(T.t(mask) + lamda)
        mask    = mask + T.arange(0,self.max_box,1)
        e_lamda = T.exp(lamda)
        lamda_x = lamda ** mask 
        fact    = T.exp(T.lgamma(T.arange(0 , self.max_box)+1))
        
        # P = ((lambda ^ x)*e^(lamda)) / x! 
        probab = (lamda_x*e_lamda)/fact
        sample = T.argmax(probab,dim=1)

        return sample