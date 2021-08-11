# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:24:51 2021

@author: Tushar and Tanishk
"""

# PATHS

SAVE_MODEL_PATH     = ""
SAVE_LOG_PATH       = ""
DATA_PATH           = ""
SAVE_OUTPUT_PATH    = ""

# Parameters
CVAE_LR         = 1e-5
BVAE_LR         = 1e-4
CVAE_EPOCHS     = 1
BVAE_EPOCHS     = 1
BVAE_LATENT_DIM = 32
N_CLASS         = 6
MAX_BOX         = 9
BVAE_BSIZE      = 256
CVAE_BSIZE      = 256
BVAE_VAL_SPLIT  = 0.1
CVAE_VAL_SPLIT  = 0.1
FRAC = 0.005

# Other
class_names = ['None' , 'Text' , 'Title' , 'List' , 'Table' ,'Figure']
