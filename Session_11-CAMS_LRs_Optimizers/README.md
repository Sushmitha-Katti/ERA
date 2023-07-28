# CAMs, LRs, and Optimizers


## Assignment
1. Check this Repo out: https://github.com/kuangliu/pytorch-cifar
2. (Optional) You are going to follow the same structure for your Code (as a reference). So Create:
	1. models folder - this is where you'll add all of your future models. Copy resnet.py into this folder, this file should only have ResNet 18/34 models. Delete Bottleneck Class
    2. main.py - from Google Colab, now onwards, this is the file that you'll import (along with the model). Your main file shall be able to take these params or you should be able to pull functions from it and then perform operations, like (including but not limited to):
		* training and test loops
		* data split between test and train
		* epochs
		* batch size
		* which optimizer to run
		* do we run a scheduler?
    3. utils.py file (or a folder later on when it expands) - this is where you will add all of your utilities like:
    	* image transforms,
		* gradcam,
		* misclassification code,
		* tensorboard related stuff
		* advanced training policies, etc
		* etc
3. Your assignment is to build the above training structure. Train ResNet18 on Cifar10 for 20 Epochs. The assignment must:
	* pull your Github code to google colab (don't copy-paste code)
	* prove that you are following the above structure
	* that the code in your google collab notebook is NOTHING.. barely anything. There should not be any function or class that you can define in your Google Colab Notebook. Everything must be imported from all of your other files
	your colab file must:
	* train resnet18 for 20 epochs on the CIFAR10 dataset
	* show loss curves for test and train datasets
	* show a gallery of 10 misclassified images
	* show gradcam output on 10 misclassified images. Remember if you are applying GradCAM on a channel that is less than 5px, then please don't bother to submit the assignment. 😡🤬🤬🤬🤬
4. Once done, upload the code to GitHub, and share the code. This readme must link to the main repo so we can read your file structure. 
5. Train for 20 epochs
6. Get 10 misclassified images
7. Get 10 GradCam outputs on any misclassified images (remember that you MUST use the library we discussed in the class)
8. Apply these transforms while training:
	* RandomCrop(32, padding=4)
	* CutOut(16x16)

## Implementation 


* Used ReduceLROnPleateau as scheduler.
* Included the plot_curve function to plot Accuracy and Loss Graphs.
* Improved GradCam Code Such that it takes misclassified images and plot the given layers. Improved the way it plots.

## Parameteres
* **Architecture** - ResNet
* **Image agumentation** - HorizontalFlip, Rotate, RGBShift, Normalise, Cutout.
* **No Of Epochs** - 20

## Results
* Best Traing Accuracy - **93.19**
* Best Test Accuracy - **90.68**
* LR Finder Curve


 ![Accuracy Change Graph](assets/accuracy_change.png "Accuracy Change Graph")

* MisClassified Images

![Misclassified with GradCAm](assets/missclassified_image.png "Misclassifed images with GradCam")

