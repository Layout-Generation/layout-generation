# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 17:44:46 2021

@author: Tushar & Tanishk
"""
import torch as T
from countvae import CountVAE
from bboxvae import BboxVAE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

########################
###### LAYOUT VAE ######
########################


class LayoutVAE(T.nn.Module):

        def __init__(self, n_class = 6, max_box = 9,bboxvae_latent_dim = 32,bboxvae_lr=1e-4,countvae_lr=1e-6):
            '''
            ** Layout VAE **
            * https://arxiv.org/abs/1907.10719
            '''
            super(LayoutVAE,self).__init__()

            self.max_box    = max_box
            self.n_class    = n_class
            self.lr_bvae    = bboxvae_lr
            self.lr_cvae    = countvae_lr
            self.countvae   = CountVAE(n_class)
            self.bboxvae    = BboxVAE(n_class,4,max_box,bboxvae_latent_dim)
            self.is_cvae_trained = 0
            self.is_bvae_trained = 0

        def forward(self,input):
            '''
            Takes only Labels Set as input
            Label Set : it is a vector of size n_class and contains 1 if correspinding class is present
            '''
            if self.is_cvae_trained == 0:
                print("[Warning] Count VAE is Not Trained !!")

            if self.is_bvae_trained == 0:
                print("[Warning] Bbox VAE is Not Trained !!")

            label_set   = input
            pred_class_counts = self.countvae(label_set , isTrain=False)

            # Normalize classiction between [0 , max_box]
            pred_class_counts = T.floor ( self.max_box*(pred_class_counts / T.sum(pred_class_counts , dim = 1 ).view(-1,1)) )

            # Extra boxes which are not be predicted
            # Their counts are set in first class
            for class_count in pred_class_counts:
                if(T.sum(class_count) < self.max_box):
                    class_count[0] = self.max_box - T.sum(class_count)

            class_labels = T.zeros(len(label_set) , self.max_box, self.n_class)

            for i in range(len(pred_class_counts)):
                l = 0
                for j in range(self.n_class):
                    for k in range(int(pred_class_counts[i][self.n_class-j-1])):
                        class_labels[i][l][self.n_class-j-1] = 1;
                        l+=1

            pred_box = self.bboxvae([ pred_class_counts, class_labels], isTrain=False)
            pred_box = pred_box.permute(2,0,1)
            class_info = T.unsqueeze(T.argmax(class_labels ,dim=2),dim=2)
            predictions = T.cat([class_info,pred_box],dim = 2)

            for i in range(len(predictions)):
                for j in range(len(predictions[i])):
                    if predictions[i][j][0]==0:
                        predictions[i][j]*=0

            self.predictions  = predictions
            self.pred_class_counts = pred_class_counts

            return predictions

        def load_data(self, path, frac = 0.5, train_test_split = 0.1):
            '''
            Loads data from npy file
            path string containig path to data
            frac defines the fraction of data to load

            '''
            try :
                Data = np.load(path)
                # Sortind Data in proper order
                np.random.shuffle(Data)
                order = np.argsort(Data[:,:,0])
                for i in range(len(Data)):
                    Data[i] = Data[i][order[i][::-1]]

                data_size = int(frac*len(Data))
                test_size = int(train_test_split*data_size)
                Data      = T.tensor(Data[0:data_size]).float()
                test_data = Data[0:test_size]
                Data      = Data[test_size:]

                # Prepare Data
                self.class_labels = Data[...,4:]
                self.class_counts = T.sum(Data[...,4:], dim = 1)
                self.b_boxes      = Data[...,0:4]
                self.label_set    = (self.class_counts !=0) + 0.0

                # Test Data
                self.test_class_labels = test_data[...,4:]
                self.test_class_counts = T.sum(test_data[...,4:], dim = 1)
                self.test_b_boxes      = test_data[...,0:4]
                self.test_label_set    = (self.test_class_counts !=0) + 0.0

                print("[Success] Data Loaded Succesfully")

            except:
                print("[Failed] Data Loading Failed\n please check path")

        def train(self, optim, train_mode = 'bboxvae', epochs = 100, bsize = 256 , validation_split = 0.1):
            '''
            * train_mode (str , default bboxvae) : Two optons
                1. if train_mode is bboxvae, BBoxVAE model will be trained and data
                will be loaded accordingly
                2. if train_mode is countvae, CountVAE model will be trained and data
                will be loaded accordingly
            * epochs (int , default 100 ) : number of epochs training should run
            * bsize(int default 256) : Batch Size
            * validation_split(float default 0.1) : should be between between 0 and 1
                1 . it defines the size of validation data

            '''
            # Create validation Split
            total_examples   = len(self.class_counts)
            val_size         = int(total_examples*validation_split)

            losses = dict()
            train_data = []
            if train_mode == 'countvae':
                model = self.countvae
                train_data = [self.label_set, self.class_counts]
            else :
                model = self.bboxvae
                train_data = [self.class_counts, self.b_boxes, self.class_labels]

            # Validation Data
            val_data = []
            for x in train_data:
                val_data.append(x[:val_size])

            # Train data
            for i in range(len(train_data)):
                train_data[i] = train_data[i][val_size:]


            # find the number of batches
            batches = len(train_data[0])//bsize
            second_loss = 'mse'
            if train_mode == 'countvae':
                second_loss = 'poisson_nll'

            # Dictionary to keep track of model statistics
            losses = {'epoch':-1,
                    'batch':0,
                    'lr' : 0,
                    'loss':0,
                    'kl_div_loss':0,
                    second_loss+'_loss':0,
                    'val_loss':0,
                    'val_kl_div_loss':0,
                    'val_'+second_loss+'_loss':0
                    }

            history  = pd.DataFrame(losses ,index = [0])
            index = 1

            for ep in range(epochs):

                # if train_mode=='countvae':
                #     self.countvae_pred_grpah(epoch = ep,path = CVAE_PATH)

                print(f'Epoch[{ep+1}/{epochs}]')
                for batch in range(batches):

                    # Get Current batch
                    b = []
                    for x in train_data:
                        b.append(x[batch*bsize : (batch+1)*bsize])

                    optim.zero_grad()

                    # Train Step
                    loss, kl_, l_ = model(b,isTrain = True)

                    # Validation Step
                    val_loss, val_kl_, val_l_ = model(val_data, isTrain = True)


                    # Save Statistics
                    losses['epoch'] = ep
                    losses['batch'] = batch
                    losses['lr']    = optim.param_groups[0]['lr']

                    loss_list = [loss, kl_, l_ , val_loss , val_kl_ , val_l_]

                    for i in range(6):
                        losses[list(losses.keys())[3+i]] = loss_list[i].cpu().clone().detach().numpy()
                        pass

                    losses_df = pd.DataFrame(losses , index=[index])
                    history   = pd.concat([history,losses_df])
                    index+=1

                    # Backpropogation step and updating weights
                    loss.backward()
                    optim.step()
                    print('\r Batch: {}/{} - loss : {} - val_loss : {} - val_{} : {}'.format(batch+1,batches,
                                                                            losses_df['loss'][index-1],
                                                                            losses_df['val_loss'][index-1],
                                                                            second_loss,
                                                                            losses_df['val_'+second_loss+'_loss'][index-1]),
                        end="")
                print("\n")
            print('[Success] Finished Training')
            return history

        def load_countvae_weights(self,path):
            try :
                self.countvae = T.load(path)
                self.is_cvae_trained=1
                print('[Success] Loaded Successfully')
            except:
                print('[Failed] Load Failed')

        def load_bboxvae_weights(self,path):
            try :
                self.bboxvae = T.load(path)
                self.is_bvae_trained=1
                print('[Success] Loaded Successfully')
            except:
                print('[Failed] Load Failed')

        def train_bboxvae(self,epochs=30, bsize=256, validation_split=0.1, optim=None):
            if optim == None:
                optim = T.optim.Adam(self.bboxvae.parameters(),lr=self.lr_bvae)

            # Start Training
            history = self.train(optim      = optim,
                            train_mode = 'bboxvae',
                            epochs     = epochs,
                            bsize      = bsize,
                            validation_split = validation_split
                        )
            self.is_bvae_trained = 1
            self.bvae_history = history[history.columns][1:]
            return self.bvae_history

        def train_countvae(self,epochs=30, bsize=256, validation_split=0.1, optim=None):

            if optim == None:
                optim = T.optim.Adam(self.countvae.parameters(),lr=self.lr_cvae)

            # Start Training
            history = self.train(optim      = optim,
                            train_mode = 'countvae',
                            epochs     = epochs,
                            bsize      = bsize,
                            validation_split = validation_split
                        )
            self.is_cvae_trained = 1
            self.cvae_history = history[history.columns][1:]
            return self.cvae_history

        def pred_countvae(self,data=None):
            '''
            * Functions is used for for predcting from CountVAE
              given label_set
            * if data is None than label set from loaded data
              are used for predictions.
            '''

            if self.is_cvae_trained == 0:
                print("[Warning] Count VAE is Not Trained !!")
            if data == None :
                data = self.test_label_set
            return self.countvae(data , isTrain=False)

        def pred_bboxvae(self,Data=None):

            '''
            * Functions is used for for predcting from BboxVAE
              given class_counts and class labels
            * if data is None than class counts and class labels from loaded data
              are used for predictions.
            '''

            if self.is_bvae_trained == 0:
                print("[Warning] Bbox VAE is Not Trained !!")

            if Data == None :
                Data = [self.test_class_counts,self.test_class_labels]

            batches = len(Data[0])//64

            for b in range(batches):

                # Get data in batch
                data = [self.test_class_counts[b*64 : (b+1)*64],
                        self.test_class_labels[b*64 : (b+1)*64]]

                # Predict
                pred = self.bboxvae(data, isTrain=False)
                pred = pred.permute(2,0,1)

                # cxywh format
                class_info = T.unsqueeze(T.argmax(data[1] ,dim=2),dim=2)
                pred = T.cat([class_info,pred],dim = 2)


                for i in range(len(pred)):
                    for j in range(len(pred[i])):
                        if pred[i][j][0]==0:
                            pred[i][j] *= 0

                if b > 0:
                    predictions = T.cat([predictions,pred],dim=0)
                else:
                    predictions = pred
            class_info =T.argmax(self.test_class_labels[0:64*batches] ,dim=2)
            class_info = T.unsqueeze(class_info,dim=2)
            gt = T.cat([class_info,self.test_b_boxes[0:64*batches]],dim = 2)
            return predictions, gt

        def countvae_pred_grpah(self,path,epoch = 0):
            pred_cvae = self.pred_countvae()
            pred_cvae = T.sum(pred_cvae,dim=0)
            pred_cvae = pred_cvae/T.sum(pred_cvae)
            pred_cvae = pred_cvae.clone().detach().numpy()

            gt_cvae = T.sum(self.class_counts,dim=0)
            gt_cvae = gt_cvae/T.sum(gt_cvae)
            gt_cvae = gt_cvae.clone().detach().numpy()

            fig   = plt.figure(figsize=(5 ,4), dpi=100 ,facecolor=(0,0,0))
            ax = fig.add_subplot()
            ax.plot(gt_cvae  , 'red',marker = 'o', label = 'Ground Truth',linewidth=4)
            ax.plot(pred_cvae,'blue',marker ='o',label = "Predicted" ,linewidth=4)
            ax.legend()
            ax.set_title('Ground Truth vs Predicted Distribution\n Epoch = '+str(epoch))
            ax.set_xlabel('Classes')
            ax.set_xticks([i for i in range(config.N_CLASS)])
            ax.set_xticklabels(config.class_names)

            plt.savefig(path+"cvae-train-ep-"+str(epoch)+".png",facecolor=(0,0,0))
            plt.close()


        def convert_to_cxywh(self,data):
            '''


            Parameters
            ----------
            data : (torch.tensor) tensor
                tensor of size (N , B , 4 + C)
                N = number of examples
                B = Number of boxes
                C = Number of classes

            Returns
            -------
            cxywh : (torch.tensor) tensor
                tensor of size (N , B , 1 + 4)
                N = number of examples
                B = Number of boxes
                c = class
                (x,y) = upper left corner
                w and h = height and width

            '''
            bboxes = data[...,0:4]
            labels = data[...,4: ]
            class_info = T.unsqueeze(T.argmax(labels ,dim=2),dim=2)
            cxywh = T.cat([class_info,bboxes],dim = 2)
            return cxywh

        def save_model(self,path):

            T.save(self.countvae,path+'countvae.h5')
            T.save(self.bboxvae,path+'bboxvae.h5')
            T.save(self,path+'layoutvae.h5')
            print('[Success] Saved Successfully')

        def save_history(self,path):

            self.cvae_history.to_csv(path+'cvae-history.csv',index=False)
            self.bvae_history.to_csv(path+'bvae-history.csv',index=False)
            print('[Success] Saved Successfully')
