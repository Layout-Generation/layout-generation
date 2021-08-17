# Layout Generation and Baseline Implementation

## 1. Layout GAN
LayoutGAN uses a GAN  network , with the generator taking randomly sampled inputs (class probabilities and geometric parameters) as parameters, arranging them and thus producing refined geometric and class parameters.

It has two kinds of discriminators mentioned, Relational discriminator which has a similar structure to the generator has effective results but cannot detect the spatial relationships well between the elements.

On the other hand Wireframe rendering discriminator uses a wireframe rendering and CNNs with a fully connected dense layer for real/fake prediction and works as a visual judge for distinguishing visual patterns.

### Generator Model architecture:
![gen_model_archi](/readme_images/generator_model.png)

### Wireframe Rendering Discriminator Model architecture: 
![wireframe_render](/readme_images/Wireframe_rendering.png)

### Results Obtained for single column layout:
![GAN_result](/readme_images/GAN_result.png)

## 2. Layout VAE
LayoutVAE is a variational autoencoder based model . It is a probabilistic and autoregressive model which generates the scene layout using latent variables in lower dimensions . It is capable of generating different layouts using the same data point.

* **CountVAE:** This is the first part of the layoutVAE model; it takes the label set as input and predicts the counts of bounding boxes for corresponding labels. The input is provided as multilabel encoding.
* **BBox VAE:** This the second part of the model was BBox VAE with LSTM based Embedding Generation. Similar to Countvae here also previous predictions along with the label set and label counts are used as conditioning info for current predictions.

### Layout VAE Model: 
![VAE_model](/readme_images/VAE_model.png)

### Flow Diagram of Both Count and BBox VAE: 
![flow_VAE](/readme_images/Flow_diag_VAE.png)

### Results Obtained:
![VAE_result](/readme_images/VAE_result.png)

## 3. Layout Transformer
Layout Transformer is a model proposed for generating structured layouts which can be used for documents, websites, apps, etc. It uses the decoder block of the Transformer Model, which is able to capture the relation of the document boxes with the previously predicted boxes (or inputs). Since it is an auto-regressive model, it can be used to generate entirely new layouts or to complete existing partial layouts.
The paper also emphasized on the fact that this model performs better than the existing models (at that time) and is better in the following aspects:
* Able to generate layouts of arbitrary lengths
* Gives better alignment due to the discretized grid
* Is able to effectively capture the relationships between boxes in a single layout, which gives meaningful layouts

### Layout Transformer Model Architecture: 
![Trans_model](/readme_images/Trans_archi.png)

### Results Obtained

![Trans_result](/readme_images/Trans_res.png)

## Quantitative Comparison:
A total of three metrics were used to compare the models. 
* Overlapping Loss
* Interection over Union (IoU)
* Alignment Loss

After Calculating the losses for each model, the following comparison table was obtained:

![metric_table](/readme_images/metric_table.png)
