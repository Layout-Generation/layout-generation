# Layout Generation and Baseline Implementation

## Contents
* [Layout VAE](#layout-vae)
  * [Layout VAE Model](#layout-vae-model)
  * [Flow Diagram of Both Count and BBox VAE](#flow-diagram)
  * [Results Obtained](#results-obtained)
* [Layout Transformer](#layout-transformer)
  * [Layout Transformer Model Architecture](#layout-transformer-model-architecture)
  * [Results](#results)
* [LayoutGAN](#layoutgan)
  * [Architecture](#architecture)
  * [Results on MNIST](#results-on-mnist)
  * [Results on single column layouts](#results-on-single-column-layouts)
* [Quantitative Comparison](#quantitative-comparison)


##  Layout VAE
LayoutVAE is a variational autoencoder based model . It is a probabilistic and autoregressive model which generates the scene layout using latent variables in lower dimensions . It is capable of generating different layouts using the same data point.

* **CountVAE:** This is the first part of the layoutVAE model; it takes the label set as input and predicts the counts of bounding boxes for corresponding labels. The input is provided as multilabel encoding.
* **BBox VAE:** This the second part of the model was BBox VAE with LSTM based Embedding Generation. Similar to Countvae here also previous predictions along with the label set and label counts are used as conditioning info for current predictions.

### Layout VAE Model 
![modelvae](https://user-images.githubusercontent.com/40228110/129761484-ba8b3494-67dc-437e-813e-705c9de19630.png)


### Flow Diagram
![Architecture](https://user-images.githubusercontent.com/40228110/129761516-a33098f9-15f1-4bcd-88de-04644beeae1c.png)


### Results Obtained
![VAE_result](/readme_images/VAE_result.png)

## Layout Transformer
Layout Transformer is a model proposed for generating structured layouts which can be used for documents, websites, apps, etc. It uses the decoder block of the Transformer Model, which is able to capture the relation of the document boxes with the previously predicted boxes (or inputs). Since it is an auto-regressive model, it can be used to generate entirely new layouts or to complete existing partial layouts.
The paper also emphasized on the fact that this model performs better than the existing models (at that time) and is better in the following aspects:
* Able to generate layouts of arbitrary lengths
* Gives better alignment due to the discretized grid
* Is able to effectively capture the relationships between boxes in a single layout, which gives meaningful layouts

### Layout Transformer Model Architecture
![Trans_model](/readme_images/Trans_archi.png)

### Results 

![Trans_result](/readme_images/Trans_res.png)

##  LayoutGAN
LayoutGAN uses a GAN  network , with the generator taking randomly sampled inputs (class probabilities and geometric parameters) as parameters, arranging them and thus producing refined geometric and class parameters.

### Architecture  
<img src="LayoutGAN/demo/layoutgan.png" width="700" height="300">

### Results on MNIST
![](LayoutGAN/demo/mnist_obtained.jpeg)

### Results on single column layouts
<img src="LayoutGAN/demo/single_col_result.png" height="787" width="473">

## Quantitative Comparison
A total of three metrics were used to compare the models. 
* Overlapping Loss
* Interection over Union (IoU)
* Alignment Loss

After Calculating the losses for each model, the following comparison table was obtained:

|                    |   Overlap   |     IOU     | Alignment |
|--------------------|:-----------:|:-----------:|:---------:|
|    Original Data   |   1.000000  |   1.000000  |  1.000000 |
|      LayoutGAN     | 1172.005234 | 2745.437529 |  1.164882 |
|      LayoutVAE     |  119.320127 |  185.864381 |  3.493406 |
| Layout Transformer |   1.090315  |   1.422297  |  0.739862 |
