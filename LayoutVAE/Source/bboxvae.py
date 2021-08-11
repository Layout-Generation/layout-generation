# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:25:31 2021

@author: Tushar & Tanishk
"""
import torch as T
from modelblocks import Encoder,Prior,EmbedBbox,ELBOLoss_Bbox
from modelblocks import ReparamatrizeMulti,Reparamatrize_cvae,Decoder

class BboxVAE(T.nn.Module):
    def __init__(self,n_class,n_dim,max_box,latent_dim=32):

        super(BboxVAE,self).__init__()
        
        self.embeder   = EmbedBbox(n_class)
        self.encoder = Encoder(n_dim,latent_dim=latent_dim)
        self.decoder = Decoder(n_dim,latent_dim=latent_dim)
        self.prior   = Prior(latent_dim=latent_dim)
        self.loss    = ELBOLoss_Bbox()
        self.rep     = Reparamatrize_cvae()
        self.n_dim   = n_dim
        self.n_class = n_class
        self.rep_mul = ReparamatrizeMulti()
        self.max_box = max_box


    def forward(self,inputs,isTrain=True):
        if isTrain==True :
            BoxCounts, GTBBox , BoxLabel= inputs
            los = 0
            kl1 = 0
            ll1 = 0
            for i in range(self.max_box):
                if i==0:
                    PrevLabel = T.zeros((1 , *BoxLabel[... ,i,:].shape)) 
                    PrevBox = T.zeros((1 , *GTBBox[...,i,:].shape))
                    

                GroundTruth = GTBBox[... , i ,:].view(-1,self.n_dim)
    
                CurrentLabel = BoxLabel[... , i ,:].view(-1,self.n_class)
    
                Embedding = self.embeder([BoxCounts,CurrentLabel,T.cat([PrevLabel,PrevBox] , dim = 2)])

                mu1 , logvar1 = self.encoder([GroundTruth,Embedding])
                mu2 , logvar2 = self.prior(Embedding)
                z1  = self.rep([mu1,logvar1])
                #z2  = self.rep([mu2,logvar2])
                
                Mu   = self.decoder([Embedding,z1])
                BBox   = self.rep_mul(Mu)
                CLoss, kl_tot , ll_tot = self.loss([mu1,logvar1,mu2,logvar2, BBox , GroundTruth])

                los = los + CLoss/self.max_box
                kl1 = kl1 + kl_tot/self.max_box
                ll1 = ll1 + ll_tot/self.max_box
                
                PrevBox = T.cat([PrevBox ,T.unsqueeze(GroundTruth,0)])
                PrevLabel = T.cat([PrevLabel , T.unsqueeze(CurrentLabel,0)])


            return los , kl1 , ll1
        else:
            BoxCounts, BoxLabel= inputs
            BBoxes = []
            for i in range(self.max_box):
                if i==0:
                    PrevLabel = T.zeros((1 , *BoxLabel[... ,i,:].shape)) 
                    PrevBox = T.zeros((1 , BoxLabel.shape[0] , 4))

                CurrentLabel = BoxLabel[... , i ,:].view(-1,self.n_class)
                Embedding = self.embeder([BoxCounts,CurrentLabel,T.cat([PrevLabel,PrevBox] , dim = 2)])
                
                mu , logvar = self.prior(Embedding)
                
                z  = self.rep([mu,logvar])
                
                Mu  = self.decoder([Embedding,z])
                
                BBox  = self.rep_mul(Mu)
                
                PrevBox = T.cat([PrevBox ,T.unsqueeze(BBox,0)])
                PrevLabel = T.cat([PrevLabel , T.unsqueeze(CurrentLabel,0)])
                BBoxes.append(BBox.t())
            BBoxes =T.stack(BBoxes)
            return BBoxes