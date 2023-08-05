
# Custom ResNet Model Training

[![Google Colab](https://img.shields.io/badge/Google_Colab-Open_In_Colab-blue?logo=google-colab)](https://github.com/Sushmitha-Katti/ERA/blob/main/Session_12_PyTorch_Lightning%20_and%20_AI%20_Application_Development/S12_final.ipynb)
[![Modular Code](https://img.shields.io/badge/Modular_Code-View_Code-green?logo=github)](https://github.com/Sushmitha-Katti/ERA/tree/main/Session_12_PyTorch_Lightning%20_and%20_AI%20_Application_Development/modular)
[![Hugging Face Gradio App](https://img.shields.io/badge/Hugging_Face-Gradio_App-blue?logo=hugging-face)](https://huggingface.co/spaces/skatti/ResNetRiddle)


This section provides an overview of the custom ResNet model architecture, training details, and performance metrics.


## Model Architecture

The custom ResNet model architecture used for this project is as follows:

- **PrepLayer:** Conv 3x3 (stride 1, padding 1) >> Batch Normalization >> ReLU [64k]
- **Layer1:** X = Conv 3x3 (stride 1, padding 1) >> MaxPooling2D >> Batch Normalization >> ReLU [128k]
  R1 = ResBlock((Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
  Add(X, R1)
- **Layer2:** Conv 3x3 [256k]
  MaxPooling2D
  Batch Normalization
  ReLU
- **Layer3:** X = Conv 3x3 (stride 1, padding 1) >> MaxPooling2D >> Batch Normalization >> ReLU [512k]
  R2 = ResBlock((Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
  Add(X, R2)
  MaxPooling with Kernel Size 4
- **Fully Connected (FC) Layer**

**Total Trainable Parameters:** 6M

## Training Details

- **Optimizer Used:** Adam
- **Loss Function:** Cross Entropy
- **Learning Rate (LR) Scheduler:** One Cycle Policy
- **Total Epochs Trained:** 24
- **Best Train Accuracy:** 98%
- **Best Test Accuracy:** 90%

## Misclassified Images

![image](https://github.com/Sushmitha-Katti/ERA/assets/36964484/f37ed89f-0bc6-4820-bf0b-6ea89efdf65d)


## Misclassified GradCam Images For 

![image](https://github.com/Sushmitha-Katti/ERA/assets/36964484/b0252e3d-2491-4843-88d8-730be665697a)

> Note: Loss and accuracy graphs can be visualized in the associated Colab notebook.



