---
title: Object Detection Using 2D Images
subtitle: "CS 496 Deep Learning Final Project: Tianfu Wang, Shangzhou Ye"
date: 2020-12-10T07:19:40.747Z
draft: false
featured: false
authors:
  - Tianfu Wang
  - Shangzhou Ye
image:
  filename: wechatimg131.jpeg
  focal_point: Smart
  preview_only: true
---
 Our project consists of detecting the 2D bounding box locations of cars using 2D images. We aim to reproduce the results for object detection using the CenterNet \cite{zhou2019objects} framework, which uses keypoint estimation to determine the center point of the object, in the KITTI data set. Our results show that the model performs fairly well in detecting the locations of cars in the KITTI dataset, achieving similar performance as what is presented in the original CenterNet paper.