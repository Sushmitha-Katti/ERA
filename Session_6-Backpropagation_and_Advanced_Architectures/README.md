# Session 6 - Backpropagation and Advanced Architectures

## Assignment - Part1

![image](https://github.com/Sushmitha-Katti/ERA/assets/36964484/2cdf5395-c805-47ac-8a69-3607189662f2)


For the above network, calculate the back propoagtion for each epoch. 
- Take a screenshot, and show that screenshot in the readme file
- The Excel file must be there for us to cross-check the image shown on readme (no image = no score)
- Explain each major step
- Show what happens to the error graph when you change the learning rate from [0.1, 0.2, 0.5, 0.8, 1.0, 2.0] 


## Solution
### Network Structure
- Inputs : 2
- Hiden Layer: 1 
- Outputs: 2
- Total Number of Weights: 8
  - Input to Hidden Layer: 4
  - Hidden Layer to Output: 4


### Backpropogation Major Steps
1. In order to get high accuaracy our total loss should be minimum. 
2. Total Loss is caclulated by sum of E1 and E2
3. For the minimum loss, we need to optimize the weights
4. So we calculate the partial derivates of total loss w.r.t all the weights
  - Each Weight(W) is updated by 
  - ```Wnew = Wold + Learning_Rate * partial derivative```
  - Learning Rate indicates the amount of change we are doing. It will be smaller value. 
6. We use activation function to achieve some non linearity.

![image](https://github.com/Sushmitha-Katti/ERA/assets/36964484/b174786f-04c4-4cd4-8f93-1589260d1a28)


### Effect of learning Rate
- As the learning rate increases, that epochs taken to reach the minimum loss increases. 
- As mentioned earlier, learning rate indicates how of learning we take from back propogation. 
- This may seem as time saving process, but increasing learning rate, we might never reach local minima. We will be jumping in between only. 
- In the particular example in excel, as the learning rate increases, the loss graph is becoming more and more non linear and our loss is decreasing faster.  

![image](https://github.com/Sushmitha-Katti/ERA/assets/36964484/dc8c8311-47c3-435e-90dc-bdde32d84acb)



## Assignment - Part2
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
