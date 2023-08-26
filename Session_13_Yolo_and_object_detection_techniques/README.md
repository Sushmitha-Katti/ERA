[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model%20Link-blue)](https://huggingface.co/spaces/skatti/YOLO-PASCO)
[![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook%20Training%20Link-blue)](https://www.kaggle.com/code/sushmithakatti/yolov3?scriptVersionId=141025965)
[![Modular Code](https://img.shields.io/badge/Modular-Code%20Link-blue)](https://github.com/Sushmitha-Katti/ERA/tree/main/Session_13_Yolo_and_object_detection_techniques/modular)

# YOLOv3 Object Detection with PyTorch Lightning

## Introduction

In this repository, we present an implementation of YOLOv3, a state-of-the-art object detection algorithm, using PyTorch Lightning. YOLOv3, short for "You Only Look Once version 3," is renowned for its real-time object detection capabilities, characterized by both speed and accuracy.

## Implementation Details

### Model

- **Model**: YOLOv3
  - Our object detection system is built upon the YOLOv3 architecture, enabling efficient and accurate object detection.

### Training

- **Trained Epochs**: 40
  - The model is trained over 30 epochs to learn and refine its object detection capabilities.

- **Batch Size**: 32
  - Training is performed with a batch size of 32 for efficient processing of data.

- **Learning Rate Scheduler**: One Cycle Policy
  - We employ the One Cycle Policy learning rate scheduler to dynamically adjust learning rates during training for improved convergence.

- **Optimizer**: Adam
  - The Adam optimizer is used to optimize the model's weights and minimize the loss function during training.

- **Initial Learning Rate**: 1E-4
  - Training begins with an initial learning rate of 0.0001.

- **Dataset**: PASCAL VOC
  - The model is trained on the PASCAL VOC dataset, a widely used dataset for object detection tasks.

- **Augmentation**: Mosaic Augmentation
  - To enhance the model's robustness, we implement Mosaic Augmentation, a special data augmentation technique that combines four random training images into a single mosaic image. This technique adds diversity to the training data and aids in improving the model's performance.

## Results

### Final Training Results

- **Train Loss**: 5.230
  - The training loss at the end of training is 5.230.

- **Class accuracy**: 78.73%
  - The model achieves an accuracy of approximately 78.73% in classifying objects.

- **No Object (No Obj) accuracy**: 99.53%
  - Impressively, the model detects the absence of objects with an accuracy of approximately 99.53%.

- **Object accuracy**: 42.66%
  - The model exhibits approximately 42.66% accuracy in detecting objects.

### Test Accuracy

- **Class accuracy**: 85.24%
  - During testing, the model's class accuracy improves to approximately 85.24%.

- **No Object (No Obj) accuracy**: 99.77%
  - The model maintains a high accuracy of approximately 99.77% in detecting the absence of objects.

- **Object accuracy**: 36.38%
  - The model's accuracy in detecting objects during testing is approximately 36.38%.
