# Session 6 - Backpropagation and Advanced Architectures


## Assignment
Write a code to train MNIST data such that it achieves the following things
- 99.4% validation accuracy
- Less than 20k Parameters
- You can use anything from above you want. 
- Less than 20 Epochs
- Have used BN, Dropout,
- (Optional): a Fully connected layer, have used GAP. 


## Solution

The CNN model consists of the following layers:

- Convolutional layers with ReLU activation and batch normalization.
- Dropout layers to prevent overfitting.
- Max pooling layer for downsampling.
- Global average pooling (GAP) layer to reduce spatial dimensions.
- Fully connected layer for classification.

### Model Summary

- Total params: 16,858
- Trainable params: 16,858
- Non-trainable params: 0
- Input size (MB): 0.00
- Forward/backward pass size (MB): 1.03
- Params size (MB): 0.06
- Estimated Total Size (MB): 1.10


### Training and Testing
- No of Epochs Trained: 20

### Results

Best Test Accuaracy : 99.47%
