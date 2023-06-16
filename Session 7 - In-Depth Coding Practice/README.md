# README

## Code Drill 1
  * Target
    * Setup Basic Skeleton. 
    * Modularize the Code
  * Result
    * Parameters: 6,379,786
    * Epochs: 20
    * Best Training Accuracy: 100
    * Best Test Accuracy: 99.38
  * Analysis
     * Extremely heavy model
     * Training Accuracy reached 100%, no more space to learn for the model
 
## Code Drill 2
  * Target 
    * Fix the Structure of Model. Going with the model where channels are increasing as we go futher. 
    * Introduce Dropout of 0.1
    * Introduce Max Pool after the Receptive Field: 5
  * Result
    * Parameters: 9,634
    * Epochs: 20
    * Best Training Accuracy: 98.80
    * Best Test Accuracy: 99.43
  * Analysis
   * Achieved the target accuracy in 19th epoch, howevers epochs and parameters constraint has not met. 
   * We have not applied any transformations yet, Still we got the better results.
   * Test Accuaracy is better than train accuracy. This means there is a scope to improve training. 

## Code Drill 3(7)
  * Target
     * Using Image Agumentation: Random Rotation
     * Reduce the number of parameters to reach the 8k parameters goal
     * Reduce the number of epochs to 15 to reach 15 epochs goal
     * Shift the max pooling after the receptive fied 7
     * Increase Dropout to 0.2
  * Result
    * Parameters: 7,674
    * Epochs: 15
    * Best Training Accuracy: 99.05
    * Best Test Accuracy: 99.41
  * Analysis
    * Reached the target once.
    * Training is still harder. Can be made slightly easier


## Code Drill 5(8)

* Target
  * Reduce the dropout to 0.1
  * In the Random Rotation fill to 0 from 1. Earlier it was 1, giving a try with 0.
* Result 
  * Parameters: 7,674
  * Best Train Accuaracy: 99.11%
  * Best Test Accuracy: 99.46%(12th Epoch)
* Analysis
  * Target Accuracy is achieved consistently. 
  * But still it is jumping between 99.4 and 99.3 from 8th epoch. This can be controlled by using suitable Learning Rate.

