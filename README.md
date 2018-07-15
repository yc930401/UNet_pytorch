# UNet for Semantic Segmentation

A UNet model for Semantic Sementation using pytorch

## Introduction

All my previous projects were built on Keras, but I found some limitations with Keras. It is easy to use predefined layer in Keras, 
but very difficult to define your own layer. Therefore I learnt Pytorch, which is very flexible. </br>
By definition, semantic segmentation is the partition of an image into coherent parts. 
For example classifying each pixel that belongs to a person, a car, a tree or any other entity in our dataset. </br>
This is the architecture of UNet: </br>
![UNet](u-net-architecture.png)
The main contribution of U-Net in this sense compared to other fully convolutional segmentation networks is that while upsampling 
and going deeper in the network we are concatenating the higher resolution features from down part with the upsampled features 
in order to better localize and learn representations with following convolutions. 
 

## Methodology

1. Prepare data for UNet (http://adas.cvc.uab.es/s2uad/?page_id=11)
2. Build a UNet model for training.
3. Train the model
4. Test

## Result
I don't have access to a powerful GPU, so the model hasn't been trained enough, but we can see that it is going in a correct direction.
You can train it for another few epochs to get better result.
Original Image: </br>
![Original Image](/data/KITTI_SEMANTIC/Result/original_000050.png)
Ground Truth: </br>
![Original Image](/data/KITTI_SEMANTIC/Result/target_000050.png)
Output: </br>
![Original Image](/data/KITTI_SEMANTIC/Result/output_000050.png)

## References:
https://github.com/A-Jacobson/Unet </br>
https://tuatini.me/practical-image-segmentation-with-unet/ </br>
https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html </br>
https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066