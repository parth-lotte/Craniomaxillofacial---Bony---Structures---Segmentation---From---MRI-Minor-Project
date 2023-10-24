
# Craniomaxillofacial Bony Structures Segmentation From MRI




## Abstract

The automated partitioning of medical images has numerous uses in clinical research. As CT (Computed Topography) imaging is essential for the diagnosis and surgical planning of craniomaxillofacial (CMF) procedures due to its ability to provide precise visualization of bone structures. Although, CT imaging leads to radiation risks and thus could hamper the subjects being scanned. On the other hand, Magnetic Resonance Imaging (MRI) generally regarded as safe compared to CT because offers effective visualization of soft tissues, although it does not provide clear visualization of bony structures. Therefore, accurately extracting bony structures from MRI images can be a demanding task. In this paper, we compare the performance of two deep learning models, FCN (Fully Convolutional Network) and U-Net for segmenting bony structures from the MRI scans using CT scans as the ground truth and achieved the 0.72 and 0.74 dice coefficient. 

## Introduction

Craniomaxillofacial Surgery (CMF) is crucial for correcting congenital facial and head conditions, necessitating precise 3D skeletal models for surgical planning. While Computed Tomography (CT) scans pose radiation risks, Magnetic Resonance Imaging (MRI) is safer but presents challenges in bone segmentation due to tissue interference and unclear bone-air boundaries. Learning-based methods, like Convolutional Neural Networks (CNNs) and Fully Convolutional Networks (FCNs), have shown promise in image segmentation. However, FCN's compression of input images can lead to information loss and accuracy issues. To address these challenges, this study compares U-Net and FCN models with various hyperparameters, evaluating their performance on augmented images for semantic segmentation in CMF surgery planning.
## Dataset

The experiments were conducted on a dataset collected from Kaggle. The dataset consists of 810 pairs of aligned **CT** and **MRI** scans of approximately 40 patients with Brain Tumor. We implemented the deep learning models using TensorFlow. The network of each model was trained on different hyperparameters. We employed the **Dice similarity** coefficient to assess the accuracy of segmentation and Intersection over union (IoU) and Dice Loss as the loss function. The performance of the models was compared before and after augmentation of the images.