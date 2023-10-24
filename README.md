
# Craniomaxillofacial Bony Structures Segmentation From MRI




## Abstract

The automated partitioning of medical images has numerous uses in clinical research. As CT (Computed Topography) imaging is essential for the diagnosis and surgical planning of craniomaxillofacial (CMF) procedures due to its ability to provide precise visualization of bone structures. Although, CT imaging leads to radiation risks and thus could hamper the subjects being scanned. On the other hand, Magnetic Resonance Imaging (MRI) generally regarded as safe compared to CT because offers effective visualization of soft tissues, although it does not provide clear visualization of bony structures. Therefore, accurately extracting bony structures from MRI images can be a demanding task. In this paper, we compare the performance of two deep learning models, FCN (Fully Convolutional Network) and U-Net for segmenting bony structures from the MRI scans using CT scans as the ground truth and achieved the 0.72 and 0.74 dice coefficient. 

## Introduction

Craniomaxillofacial Surgery (CMF) is crucial for correcting congenital facial and head conditions, necessitating precise 3D skeletal models for surgical planning. While Computed Tomography (CT) scans pose radiation risks, Magnetic Resonance Imaging (MRI) is safer but presents challenges in bone segmentation due to tissue interference and unclear bone-air boundaries. Learning-based methods, like Convolutional Neural Networks (CNNs) and Fully Convolutional Networks (FCNs), have shown promise in image segmentation. However, FCN's compression of input images can lead to information loss and accuracy issues. To address these challenges, this study compares U-Net and FCN models with various hyperparameters, evaluating their performance on augmented images for semantic segmentation in CMF surgery planning.
## Dataset

The experiments were conducted on a dataset collected from Kaggle. The dataset consists of 810 pairs of aligned **CT** and **MRI** scans of approximately 40 patients with Brain Tumor. We implemented the deep learning models using TensorFlow. The network of each model was trained on different hyperparameters. We employed the **Dice similarity** coefficient to assess the accuracy of segmentation and Intersection over union (IoU) and Dice Loss as the loss function. The performance of the models was compared before and after augmentation of the images.

# MRI- Scan 
![alt text](https://github.com/parth-lotte/Craniomaxillofacial---Bony---Structures---Segmentation---From---MRI-Minor-Project/blob/master/mri_0.png)                                                                          ![alt text](https://github.com/parth-lotte/Craniomaxillofacial---Bony---Structures---Segmentation---From---MRI-Minor-Project/blob/master/mri_1.png)

# CT-Scan
![alt text](https://github.com/parth-lotte/Craniomaxillofacial---Bony---Structures---Segmentation---From---MRI-Minor-Project/blob/master/ct_0.png)                                                                          ![alt text](https://github.com/parth-lotte/Craniomaxillofacial---Bony---Structures---Segmentation---From---MRI-Minor-Project/blob/master/ct_1.png)   


# Model

 **FCN**

The FCN model we implemented has an encoder consisting of three blocks of two convolutional layers followed by a max-pooling layer to reduce the spatial dimension of the feature maps as shown in figure 2. Each block starts with a convolutional layer with 64, 128, and 256 filters, respectively, with a 3x3 kernel and ReLU activation function [15]. The bottleneck layer consists of a single convolutional layer with 512 filters of size 3x3 and a ReLU activation function. The decoder consists of three blocks of two convolutional layers followed by an up-sampling layer to increase the spatial resolution of the feature maps. Each block starts with an up-sampling layer followed by a convolutional layer with 256, 128, and 64 filters, respectively, with a 3x3 kernel and ReLU activation function. The final layer is a convolutional layer with a 1x1 kernel and sigmoid activation function.

![alt text](https://github.com/parth-lotte/Craniomaxillofacial---Bony---Structures---Segmentation---From---MRI-Minor-Project/blob/master/new_alex-model.jpg)



**U-NET**

The U-Net architecture is a fully convolutional network that is represented in a 'U' shape, consisting of an encoder on the left and a decoder on the right as shown in figure 3. The input image is passed through a series of convolutional layers with ReLU activation function, resulting in a reduction of image size [16]. The encoder block includes max-pooling layers with increasing numbers of filters, while the decoder block has decreasing numbers of filters for gradual upsampling of the image. Skip connections are used to connect the previous outputs with the decoder layers to preserve notable features and lead to faster convergence. The final convolution block consists of convolutional layers and a filter with the appropriate function to produce the output.

![alt text](https://github.com/parth-lotte/Craniomaxillofacial---Bony---Structures---Segmentation---From---MRI-Minor-Project/blob/master/U%20Net.png)


