
# Custom ResNet Model Training

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

## Additional Resources

- [Link to Misclassified Images](link-to-misclassified-images)
- [Google Colab Link for Training](google-colab-link)
- [Modular Code for Model Training](link-to-modular-code)


> Note: Loss and accuracy graphs can be visualized in the associated Colab notebook.

## Grad-CAM Images

[Link to Grad-CAM Images](link-to-gradcam-images)

