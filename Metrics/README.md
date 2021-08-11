# Metrics/Quantitative Comparison
## Intersection over Union (IoU)
The intersection over the union of boxes is calculated pairwise and are then added together. The overall IoU of the data is averaged over all the documents.

For the kth document in the data, the iou Lk is calculated as follows:

![iou1](/readme_images/iou1.jpg)

Where n is the total number of boxes in the document.

For the whole data, the loss(IoU) is calculated as follows:

![iou2](/readme_images/iou2.jpg)

Where N is the total number of documents in the data.


## Overlapping Loss
Overlapping loss is defined as the ratio of overlapping area by the box area. It is also calculated pairwise, added together and then averaged for all documents. Related expressions are given below:

![overlapping1](/readme_images/overlapping1.jpg)

![overlapping2](/readme_images/iou2.jpg)

## Alignment Loss
Adjacent elements (boxes) are usually in six possible alignment types: Left, X-center, Right, Top, Y-center and Bottom aligned. Denote =(xL,yT,xC,yC,xR,yB) as the top-left, center and bottom-right coordinates of the predicted bounding box, we encourage pairwise alignment among elements by introducing an alignment loss:

![alg1](/readme_images/algn1.jpg)

![alg2](/readme_images/algn2.jpg)

## Comparison
![comp](/readme_images/comp.png)
