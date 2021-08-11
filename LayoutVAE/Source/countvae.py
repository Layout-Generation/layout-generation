# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:11:05 2021

@author: Tushar & Tanishk
"""
import torch as T
from torch.distributions import  Poisson
from modelblocks import Encoder, Decoder, Prior, Embeder, ELBOLoss, Reparamatrize_cvae, Sampling 
 
class CountVAE(T.nn.Module):
 
    def __init__(self,n_class,max_box=9):
        '''
        n_class = number of class (intger)
        max_box = maximum number of boxes (integer)
        isTrain(boolean) default False : defines whether data is to be treated as training data or testing
        
        if isTrain = True :
            input must be a tuple with first value corresponding to label set and second corresponding to ground Truth
            counts
        else :
            input must have label set
        
        '''
        super(CountVAE,self).__init__()
        
        
        self.encoder = Encoder()
        self.prior   = Prior()
        self.decoder = Decoder(1)
        self.embeder = Embeder(n_class)
        self.loss    = ELBOLoss()  
        self.rep     = Reparamatrize_cvae()
        self.n_class = n_class
        self.pois    = Sampling(max_box)
                
    def forward(self, inputs, isTrain = False):
        
        if isTrain==True:
            
            label_set , groundtruth_counts = inputs
            Loss = 0
            LL   = 0
            KL   = 0
            previous_counts = T.zeros_like(label_set)
            
            for i in range(self.n_class):
            
                current_label = T.zeros_like(previous_counts)
                x_ = label_set[...,i]
                current_label[...,i]= x_
                z_ = groundtruth_counts[...,i].view(-1,1)
                
                # Generate Conditional Embedding
                embedding    = self.embeder([label_set, current_label, previous_counts])
                
                # Encoding To latet space
                mu1, logvar1 = self.encoder([z_,embedding])
                mu2, logvar2 = self.prior(embedding)
                
                # Reparamatrized Latent variable
                z  = self.rep([mu1,logvar1])

                # Decode from Latent space
                decoded = self.decoder([embedding,z])
                Closs, L_, kl_ = self.loss([mu1, logvar1, mu2, logvar2, decoded , z_])
                
                # Update Losses
                Loss   = Loss + Closs
                LL     = LL   + L_
                KL     = KL   + kl_
                
                decoded = T.exp(decoded)
                
                # Poisson Distributions with rate of Deoded
                # q = self.pois(decoded)
                q = Poisson(decoded).sample()
                
                # update Preivious Counts
                previous_counts = previous_counts + current_label*(q.view(-1,1) +  x_.view(-1,1))
            
            return  Loss/self.n_class, KL/self.n_class, LL/self.n_class
        
        else:
            
            label_set = inputs
            previous_counts = T.zeros_like(label_set)
            
            for i in range(self.n_class):

                current_label = T.zeros_like(previous_counts)
                x_ = label_set[...,i]
                current_label[...,i]= x_
                
                
                # Generate Conditional Embedding
                embedding = self.embeder([label_set, current_label, previous_counts])
                
                # Encoding To latet space
                mu,logvar = self.prior(embedding)
                
                # Reparamatrized Latent variable
                z = self.rep([mu,logvar])
                
                # Decode from Latent space
                decoded = self.decoder([embedding,z])
                decoded = T.exp(decoded)

                # Poisson Distributions with rate of Deoded
                # q = self.pois(decoded)
                q = Poisson(decoded).sample()
                
                 # update Preivious Counts
                previous_counts = previous_counts + current_label*(q.view(-1,1) +  x_.view(-1,1))
                
            return previous_counts