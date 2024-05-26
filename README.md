# Deep-Learning

## Overview

This repository contains pre-trained models and scripts for age, gender, and emotion classification using deep learning techniques. 

The models for age and gender classification are based on the Caffe framework, while the emotion detection model is custom-built from scratch.

## Files

### Models

emotion_detection.h5: Custom pretrained model for emotion detection, developed from scratch

age_net.caffemodel: Pretrained model for age classification using the Caffe framework.

gender_net.caffemodel: Pretrained model for gender classification using the Caffe framework.

### Network Definitions

age_deploy.prototxt: Network definition for the age classification model.

gender_deploy.prototxt: Network definition for the gender classification model.

### Scripts

facial_detector.py: Python script that uses OpenCV to detect faces in images and prepares them for classification.

### Notebooks

emotion_detection.ipynb: Jupyter notebook for demonstrating emotion detection using the custom model. It includes steps for loading the model, preprocessing input images, and making predictions.

### Supporting Files

haarcascade_frontalface_default.xml: Haarcascade file for detecting faces in images using OpenCV.

### Acknowledgements

Special thanks to the creators of the original age and gender models and the Caffe framework.
