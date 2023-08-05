# imports
import os
import math


import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms,datasets
from torchvision.datasets import MNIST
from torchsummary import summary
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import pytorch_lightning as pl






classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




os.environ['KMP_DUPLICATE_LIB_OK']='True'


# DataSet for CiFAR 10
class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label



# MisclassifiedImageLogger Callback
class MisclassifiedImageLogger(pl.Callback):
    def __init__(self, classes, max_images=10):
        self.classes = classes
        self.max_images = max_images
        self.misclassified_images = []
        self.misclassified_labels = []
        self.predicted_labels = []

    # collect the misclassified data
    def on_test_end(self, trainer, pl_module):
        self.data_loader = pl_module.test_dataloader()
        self.model = trainer.model

        print(trainer.model.device)

        self.model.eval()
        for batch in self.data_loader:
            inputs, labels = batch
            inputs = inputs.to( trainer.model.device)
            labels = labels.to(trainer.model.device)
            outputs = self.model(inputs)

            _, preds = torch.max(outputs, 1)
            misclassified_mask = preds != labels

            self.misclassified_images.extend(inputs[misclassified_mask])
            self.misclassified_labels.extend(labels[misclassified_mask])
            self.predicted_labels.extend(preds[misclassified_mask])


        self.plot_misclassified(self.misclassified_images, self.misclassified_labels, self.predicted_labels)

    # plot the misclasified data
    def plot_misclassified(self, images, true_labels, predicted_labels):
        plt.figure(figsize=(12, 6))
        for i in range(min(self.max_images, len(images))):
            plt.subplot(2, 5, i + 1)
            image = images[i].cpu().numpy().transpose((1, 2, 0))
            true_label = self.classes[true_labels[i]]
            predicted_label = self.classes[predicted_labels[i]]
            plt.imshow(image)
            plt.title(f"True: {true_label}\nPredicted: {predicted_label}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


# Display grad cam impages
def display_gradcam_output(data: list,
classes: list[str],
inv_normalize: transforms.Normalize,
model,
target_layers,
targets=None,
number_of_samples: int = 10,
transparency: float = 0.60):
  """
  Function to visualize GradCam output on the data
  :param data: List[Tuple(image, label)]
  :param classes: Name of classes in the dataset
  :param inv_normalize: Mean and Standard deviation values of the dataset
  :param model: Model architecture
  :param target_layers: Layers on which GradCam should be executed
  :param targets: Classes to be focused on for GradCam
  :param number_of_samples: Number of images to print
  :param transparency: Weight of Normal image when mixed with activations
  """
  # Plot configuration
  fig = plt.figure(figsize=(10, 10))
  x_count = 5
  y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples /
  x_count)
  # Create an object for GradCam
  cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
  # Iterate over number of specified images
  for i in range(number_of_samples):
    plt.subplot(y_count, x_count, i + 1)
    input_tensor = data[i][0]
    cloned_input_tensor = input_tensor.clone().detach().unsqueeze(0)
    # Get the activations of the layer for the images
    grayscale_cam = cam(input_tensor=input_tensor.to('cpu').unsqueeze(0), targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    # Get back the original image
    img = input_tensor.squeeze(0).to('cpu')
    img = inv_normalize(img)
    rgb_img = np.transpose(img, (1, 2, 0))
    rgb_img = rgb_img.numpy()
    # Mix the activations on the original image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True,
    image_weight=transparency)
    # Display the images on the plot
    plt.imshow(visualization)
    plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' +
    classes[data[i][2].item()])
    plt.xticks([])
    plt.yticks([])