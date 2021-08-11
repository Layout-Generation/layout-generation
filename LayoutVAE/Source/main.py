# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:26:19 2021

@author: Tushar & Tanishk
"""
from layoutvae import LayoutVAE
from utils import plot_layouts ,plot_history,generate_colors,countvae_pred_graph
import config


# Model
layoutvae = LayoutVAE(n_class=config.N_CLASS,   
                      max_box=config.MAX_BOX,
                      bboxvae_latent_dim=config.BVAE_LATENT_DIM,
                      bboxvae_lr=config.BVAE_LR,
                      countvae_lr=config.CVAE_LR,
                      )
layoutvae.load_data(path = config.DATA_PATH,frac= config.FRAC)

history_bvae_df  = layoutvae.train_bboxvae(bsize = config.BVAE_BSIZE,
                                           epochs=config.BVAE_EPOCHS,
                                           validation_split=config.BVAE_VAL_SPLIT)


history_cvae_df  = layoutvae.train_countvae(bsize = config.CVAE_BSIZE,
                                           epochs=config.CVAE_EPOCHS,
                                           validation_split=config.CVAE_VAL_SPLIT)

# Save History
layoutvae.save_model(config.SAVE_MODEL_PATH)
layoutvae.save_history(config.SAVE_LOG_PATH)

# Predict Layout
colors = generate_colors(n_class=config.N_CLASS,
                         class_names=config.class_names)

# only using bboxvae
pred , ground_truth = layoutvae.pred_bboxvae()
for i in range(2):
    plot_layouts(pred = pred[i*16:(i+1)*16],
                 colors=colors,
                 class_names=config.class_names,
                 path=config.SAVE_OUTPUT_PATH+"bvae-preds-"+str(i)+".png"
                 )
    
# using complete model
final_predictions = layoutvae(layoutvae.label_set)

#visualize and save predictions


plot_layouts(pred = final_predictions,
            colors = colors,
            title = "Random Outputs",
            class_names=config.class_names,
            path = config.SAVE_OUTPUT_PATH+"randout.svg")

countvae_pred_graph(layoutvae,config.SAVE_OUTPUT_PATH+"cvae-train.png")

plot_layouts(pred = pred,
            colors = colors,
            title = "BBoxVAE Outputs",
            class_names=config.class_names,
            path = config.SAVE_OUTPUT_PATH+"bboxvae-pred.svg")

# Plot and save Train History plots
plot_history(layoutvae.bvae_history , path = config.SAVE_LOG_PATH+"bvae-train.png")
plot_history(layoutvae.cvae_history , path = config.SAVE_LOG_PATH+"cvae-train.png")

# Complete Model
predd = layoutvae(layoutvae.test_label_set)
for i in range(2):
    plot_layouts(pred = predd[i*16:(i+1)*16],
                 colors=colors,
                 class_names=config.class_names,
                 path=config.SAVE_OUTPUT_PATH+"random-preds-"+str(i)+".png"
                 )
