# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:36:04 2021

@author: Tushar & Tanishk
"""

import torch as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import config
plt.style.use('dark_background')

def plot_history(history,title = 'Training Statistics', path =""):
    height = 12
    width  = 9
    fig          = plt.figure(figsize=(width,height), dpi=100 ,facecolor=(0,0,0))
    height_ratio = [0.25,1,1,1]
    grid         = plt.GridSpec(4,2,
                        hspace=0.3,wspace=0.2,
                        height_ratios =height_ratio,
                        left=0.02,right=0.98,top=0.98,bottom=0.02
                    )
    index = 0
    ax = fig.add_subplot(grid[index : index+2])
    index+=2
    ax.text(x = 0.3 ,y = 0.5 ,s = title,fontsize=30)
    ax.invert_yaxis()
    ax.axis('off')
    colors = ['red','blue','green']
    for i in range(3):

        ax = fig.add_subplot(grid[index])
        ax.plot(history[history.columns[i+3]],colors[i])
        index+=1
        ax.set_facecolor((0,0,0))
        ax.set_title(history.columns[i+3])
        ax = fig.add_subplot(grid[index])
        ax.plot(history[history.columns[i+6]],colors[i])
        ax.set_title(history.columns[i+6])
        index+=1
        ax.set_facecolor((0,0,0))
        ax.set_xlabel('Batches')
        ax.set_ylabel('Loss')
    plt.savefig(path, facecolor=(0,0,0))


def generate_colors(class_names = None,n_class=6):
    '''

    Parameters
    ----------
    class_names : list, optional
        List of classes in the dataset. The default is None.
    n_class : integer, optional
        The default is 6.
    Returns
    -------
    colors : list of hexadecimal strings
    
    '''
    cmap = ["","#dc143c","#ffff00","#00ff00","#ff00ff","#1e90ff","#fff5ee",
            "#00ffff","#8b008b","#ff4500","#8b4513","#808000","#483d8b",
            "#008000","#000080","#9acd32","#ffa500","#ba55d3","#00fa9a",
            "#dc143c","#0000ff","#f08080","#f0e68c","#dda0dd","#ff1493"]
            
    colors = dict()

    if class_names == None:
        class_names = []
        for i in range(n_class):
            class_names.append('class'+str(i+1))
    
    for i in range(n_class):
        colors[class_names[i]] = cmap[i]

    return colors

class_names = ['None' , 'Text' , 'Title' , 'List' , 'Table' ,'Figure']
colors = generate_colors(n_class=6 , class_names=class_names)

def plot_layouts(pred,colors,class_names,title="Predictions", path=""):
    '''
    data in cxywh format
    '''
    height = 15
    width  = 9
    fig          = plt.figure(figsize=(width,height), dpi=50 ,facecolor=(0,0,0))
    height_ratio = [0.25,0.25,1,1,1,1]
    grid         = plt.GridSpec(6,4,
                        hspace=0.05,wspace=0.05,
                        height_ratios =height_ratio,
                        left=0.02,right=0.98,top=0.98,bottom=0.02
                    )
    index = 0


    ax = fig.add_subplot(grid[index : index+4])
    index+=4
    ax.text(x = 0.2 ,y = 0.5 ,s = title,fontsize=30)

    legend = []
    ax = fig.add_subplot(grid[index : index+4])
    index += 4
    
    for i in range(1,6):
        legend.append(Patch(facecolor=colors[class_names[i]]+"40",
                            edgecolor=colors[class_names[i]],
                            label= class_names[i]))
        
    ax.legend(handles=legend, ncol=3,loc=8, fontsize=25, facecolor=(0,0,0))
    ax.axis('off')

    for i in range(16):
        ax   = fig.add_subplot(grid[index])
        index += 1
        
        data = pred[i]
        rect1 = patches.Rectangle((0,0),180,240)
        rect1.set_color((0,0,0,1))
        ax.add_patch(rect1)
        for box in data:

            c,x,y,w,h = box
            if c==0:
                continue
            x = x*180
            y = y*240
            w = w*180
            h = h*240
            rect = patches.Rectangle((x,y),w,h,linewidth=2)
            rect.set_color(colors[class_names[int(c)]]+"72")
            rect.set_linestyle('-')
            rect.set_edgecolor(colors[class_names[int(c)]])
            ax.add_patch(rect)
        ax.plot()
        ax.set_facecolor((0,0,0))
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(2)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(path , facecolor=(0,0,0))

def countvae_pred_graph(model,path=""):
            pred_cvae = model.pred_countvae()
            pred_cvae = T.sum(pred_cvae,dim=0)
            pred_cvae = pred_cvae/T.sum(pred_cvae)
            pred_cvae = pred_cvae.to('cpu').clone().detach().numpy()

            gt_cvae = T.sum(model.class_counts,dim=0)
            gt_cvae = gt_cvae/T.sum(gt_cvae)
            gt_cvae = gt_cvae.to('cpu').clone().detach().numpy()

            fig   = plt.figure(figsize=(5 ,4), dpi=100 ,facecolor=(0,0,0))
            ax = fig.add_subplot()
            ax.plot(gt_cvae  , 'red',marker = 'o', label = 'Ground Truth',linewidth=4)
            ax.plot(pred_cvae,'blue',marker ='o',label = "Predicted" ,linewidth=4)
            ax.legend()
            ax.set_title('Ground Truth vs Predicted Distribution')
            ax.set_xlabel('Classes')
            ax.set_xticks([i for i in range(config.N_CLASS)])
            ax.set_xticklabels(config.class_names)

            plt.savefig(path,facecolor=(0,0,0))
            plt.close()