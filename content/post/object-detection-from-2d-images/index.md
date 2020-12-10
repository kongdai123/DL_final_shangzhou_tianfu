---
title: 2D Object Detection on KITTI
subtitle: ""
date: 2020-12-10T08:27:05.570Z
draft: false
featured: false
authors:
  - Tianfu Wang
  - Shangzhou Ye
image:
  filename: featured
  focal_point: Smart
  preview_only: false
---
*Northwestern University, Fall 2020 CS 496 Deep Learning Final Project.*

*Instructor: Prof. Bryan Pardo.*

*Team Members: Tianfu Wang (tianfuwang2021@u.northwestern.edu), Shangzhou Ye (shangzhouye2020@u.northwestern.edu)*



![Bounding box detection result of a example image.](000325.jpg "Bounding box detection result of an image in the KITTI dataset.")

<p style="text-align: center;"><sub><sup>Bounding box detection result of an image in the KITTI dataset</sup></sub></p>



## Abstract

We implemented CenterNet \[1] from scratch for 2D object detection and tested on [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). We aimed to reproduce the results as what is presented in the original [CenterNet](https://arxiv.org/pdf/1904.07850.pdf) paper. The model represents each object as a single point - the center point of the 2D bounding box. DLA-34 \[2] is used as our backbone for center point estimation, and other object properties including width and length of the bounding box are regressed from the center point. We achieved 92.10% AP for easy objects, 86.72% AP for moderate objects and 78.73% AP for hard objects respectively. The performance of our implementation is similar to the original CenterNet paper.



Our code is on <https://github.com/shangzhouye/centernet-detection-kitti>. 

Our paper is at this [link](https://drive.google.com/file/d/1UsQk9BLiQ60QheyG00QhtySjKP13MJG9/view?usp=sharing).



## The Problem



The goal of our final project is to train a model that accurately detects the 2D location of cars purely 2D images. We use the KITTI dataset to predict the object’s bounding box location in the image pixel space. We aim to reproduce the result from the CenterNet paper.



## Introduction and motivation



Object detection from 2D images is a very relevant and rapidly advancing problem as of today. It has many applications in the development of autonomous systems, and most notably, self-driving cars. In order to build automation system that properly and responsively react to its surroundings, it needs a vision system that achieves object detection that is accurate, fast, and robust to change in relative positions, viewing angles, lighting, and occlusions. Humans instantly recognize their surrounding objects and  can  estimate the position of objects without any explicit measurement. Therefore, it is intellectually interesting and practical to investigate how a deep learning model would perform in the task of detecting objects from 2D images.



## Previous Work



One of the first successful attempts to use a deep network in object detection is RCNN\[3], which crops certain region candidates of the image and apply classification through a deep network. However, such methods are very slow since the set of possible candidate region is very large to enumerate.



Keypoint estimation is a technique in used mainly in human pose estimation to detect the location of human joint locations. One popular approach is the stacked hourglass model \cite{newell2016stacked}, which used repeated bottom-up, top-down  the successive steps of pooling and upsampling that captures the image information on multiple scales to achieve keypoint detection. 



Before CenterNet, keypoint estimation already been used in the task of object detection. Models such as ExtremeNet\cite{zhou2019bottom} detects the four vertices as well as the center of the bounding box. However, after the keypoint estimation stage, the detected keypoints needs to be grouped to form the final bounding box, and this grouping process is very slow. 

## Network Design



The CenterNet framework \[1] models the object as a single point, which is the center of the bounding box in the image. CenterNet first uses keypoint estimation to find center points. The image is fed into a fully-convolutional encoder-decoder network, and the output is a heatmap for each class with values between \[0,1]. Peaks in the heatmap correspond to object centers. In our project, we use a DLA-34 network \[2] as the backbone for our keypoint estimation system. For our training, the input is the KITTI data set image resized to 512 * 512 pixels. We then calculate the center position p of the car objects in the resized image space from the label data, and generate the ground truth heatmap by passing the center keypoint though a Gaussian smoothing kernel, where the intensity value of each pixel $I(x,y) = exp(\frac{-((x - p*x)^2 + (y - p_y)^2)}{2\sigma_p^2})$ ($\sigma_p$ is adaptive to the object size). A pixel-wise maximum is taken should two Gaussians overlap. A penalty-reduced pixel-wise logistic regression with focal loss is then used for the training.
$$L = -\frac{1}{N}\sum*{x,y} \begin{cases} 
      (1 - \Tilde{I}(x,y))^{\alpha}\log(1 - \Tilde{I}(x,y))  & I(x,y) = 1 \
      (1 - I(x,y))^{\beta} (\Tilde{I}(x,y))^{\alpha}\log(1 - \Tilde{I}(x,y)) & otherwise
   \end{cases}
$$
$N$ is the number of keypoints, and $\alpha, \beta$ are hyper-parameters for the focal loss \cite{lin2017focal}.

Once the keypoint detection heatmap is generated, other properties, such as the bounding box of the object, are then regressed from the image features at the center location. The regression shares the same fully-convolutional backbone  with  the  keypoint  estimator with a separate regression head for each property. The loss function for the regression is the L2 loss between the predicted size of the bounding box and the ground truth size of the  bounding box.



## Training and Testing



We use the KITTI \cite{geiger2012we} Vision Benchmark Suite. The dataset is already labeled and has a size of 24 GB. The KITTI dataset is compiled for autonomous driving development. The images of the KITTI dataset consist of mainly outdoor roads scenes, with a lot of cars and other objects like pedestrians and houses. It consists of 7481 training images and 7518 test images, comprising a total of 80256 labeled objects. For this project, we focus on object detection for cars only. Because only those 7481 training images have publicly available labels, we random split them into training and validation sets. The training set is 80% of the whole dataset (5984 images) while the validation is 20% of the whole dataset (1497 images). No data augmentation is utilized for our project.

The data consists of 2D RGB images and a corresponding $txt$ file for the labels. In the label $txt$ file, there are 16 values separated by spaces for each labeled object. The first value is the class in string format, the second value is the “truncated” value, a 0 or 1 value that refers to the object being on the edge of an image. The third value is the “occluded” value, a 0 or 1 value which refers to the object being occluded. The 4th value is the alpha, which is the observation angle of the object, ranging from $-\pi$ to $\pi$. The 5th to 8th value is the 2D bounding box. The next 9th to 11th values are the dimensions height, width and length in meters. The next 12th to 14th values are the x,y,z location in meters. The 15th value is the rotation value, the angle of the object with respect to the camera in $\[-\pi, \pi]$. Finally, the 16th value is the score, only for predictions, ranging from 0 to 1. We output our prediction in the same format as the labels for evaluation.

For the evaluation, we followed the standard average precision (AP) evaluation criteria proposed in the Pascal VOC benchmark \cite{EveringhamMark2009TPVO}. A car detection can be counted as true positive only if its overlap with the ground truth bounding box is above 70%. By adjusting the confidence threshold for detection, a precision-recall (PR) curve can be obtained with 40 different recall positions. The AP can then be calculated as the area under the PR curve. We use this calculated average precision value as the measure of the performance of our system. The KITTI benchmark evaluation criterion has three levels of difficulty: Easy, Medium, and Hard  \cite{geiger2012we}. The object's minimum bounding box height decreases with increasing difficulty, while the maximum occlusion level and maximum truncation increases with increasing difficulty.



## Results



**Implementation details:** We use 34-layer deep layer aggregation (DLA) network \cite{YuFisher2017DLA} as our backbone. The heatmap from keypoint estimator has the size of $128 \times 128$ with an output stride of 4. There is an additional local offset prediction to compensate the decrease in resolution. The weight of heatmap loss, width/height loss and offset loss are 1, 0.1 and 0.1 respectively. We trained with batch-size of 8 (on 1 GPU) and learning rate of 5e-4. The models converges after 3 epochs and start to over-fitting after that.

![](screenshot-from-2020-12-10-15-52-19.png)

Table \ref{tab:result} shows our evaluation results compared to the original CenterNet paper. It shows that our implementation is able to achieve similar performance as the original paper. Notice that the original paper follows a 50/50 training and validation split and we are having an 80/20 split. Also, the results of the original paper is based on all classes but we only focused on cars predictions in this project.

![](combined.jpg)

![](heatmap_compare.jpg)

Figure \ref{fig:example} shows an example inference result compared to the ground truth. It is shown that our model to able to predict most of the objects correctly in this scene\footnote{More examples can be found on our website.}. Figure \ref{fig:heatmap} shows the comparison between the ground truth heatmap with Gaussian smoothing and our predicted heatmap on the same image.

![](pr_curve.png)

Figure \ref{fig:pr} shows the precision-recall curve of our final model on the validation set. Three curves represent easy, moderate and hard objects respectively. The area under the curve is the average precision.



## Future Work



One of the main advantages of the CenterNet architecture is that it can be very easily extended to other tasks, such as 3D detection, as well as human pose estimation, with minor effort. Once the heat map for center detection is obtained, more properties of the image can be learned simply by changing the regression head of the model. It would be very interesting to see how the model performs when detecting 3D location of cars without any explicit depth measurement like LiDAR. Due to the short time frame of this project, we are unable to get to the point of doing 3D detection, but it is certainly a intriguing direction to take further on.