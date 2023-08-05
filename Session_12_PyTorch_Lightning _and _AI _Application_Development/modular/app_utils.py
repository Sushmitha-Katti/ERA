from PIL import Image

import numpy as np
import gradio as gr
from pytorch_grad_cam import GradCAM
from torchvision import transforms
import torch


from utils import show_cam_on_image
from CustomResnet import CustomResNet
from utils import Cifar10SearchDataset, MisclassifiedImageLogger, display_gradcam_output




model = CustomResNet.load_from_checkpoint('custom_resnet_model.ckpt',map_location=torch.device('cpu') )


# Denormalize the data using test mean and std deviation
inv_normalize = transforms.Normalize(
    mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
    std=[1/0.23, 1/0.23, 1/0.23]
)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --------------------------Show GradCam Images ---------------------------------------
def show_gradcam_images(model, target_layer, transparency, input_img, use_cuda= False):
  target_layer_map = {
      "layer_1" : model.layer_1,
      "layer_2": model.layer_2,
      "layer_3" : model.layer_3,
  }
  cam = GradCAM(model=model, target_layers=[target_layer_map[target_layer]], use_cuda=use_cuda)
  grayscale_cam = cam(input_tensor=input_img, targets=None)
  grayscale_cam = grayscale_cam[0, :]
  img = input_img.squeeze(0).to('cpu')
  img = inv_normalize(img)
  rgb_img = np.transpose(img, (1, 2, 0))
  rgb_img = rgb_img.numpy()
  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)
  return visualization


# --------------------------Inference ---------------------------------------
def inference(input_img, transparency, top_classes=1, grad_cam_layer_list= ['layer_1']):
    if top_classes < 1 or top_classes > 10:
      gr.Error("Top Classes should be between 1-10")
    transform = transforms.ToTensor()
    input_img = transform(input_img)
    input_img = input_img.to('cpu')
    input_img = input_img.unsqueeze(0)
    outputs = model(input_img)
    softmax = torch.nn.Softmax(dim=0)
    o = softmax(outputs.flatten())
    softmax_values, indices = torch.sort(o,descending=True)
    softmax_values = softmax_values[:int(top_classes)]
    indices = indices[:int(top_classes)]
    confidences = {}
    confidences = {classes[index]: float(softmax_values[i]) for i, index in enumerate(indices)}
    _, prediction = torch.max(outputs, 1)
    visualizations = []
    for layer in grad_cam_layer_list:
      visualization = show_gradcam_images(model, layer, transparency, input_img)
      visualizations.append((visualization, layer))
    return confidences, visualizations



# ---------------------------------------- Get Sample MisClassified Images ---------------------------------------
def get_misclassified_images(misclassifications):
   misclassified_images = [    
      ("misclassified_images/misclassified_1.png", ""),
      ("misclassified_images/misclassified_2.png",""),
      ("misclassified_images/misclassified_3.png",""),
      ("misclassified_images/misclassified_4.png",""),
      ("misclassified_images/misclassified_5.png",""),
      ("misclassified_images/misclassified_6.png",""),
      ("misclassified_images/misclassified_7.png",""),
      ("misclassified_images/misclassified_9.png",""),
      ("misclassified_images/misclassified_9.png",""),
      ("misclassified_images/misclassified_10.png","")
   ]
   return misclassified_images[:int(misclassifications)]
  



# ---------------------------------------- Get Sample MisClassified Images ---------------------------------------
def get_gradcam_images( gradcam_number, layers=['layer_1']):

   empty_image = ('testfigure.png', "")
   

   layer_1_gradcam_images = [
      ("gradcam/gradcam_layer_1_1.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_2.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_3.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_4.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_5.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_6.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_7.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_8.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_9.png",  "layer_1"),
      ("gradcam/gradcam_layer_1_10.png", "layer_1"),
   ]
   layer_2_gradcam_images = [
      ("gradcam/gradcam_layer_2_1.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_2.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_3.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_4.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_5.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_6.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_7.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_8.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_9.png",  "layer_2"),
      ("gradcam/gradcam_layer_2_10.png", "layer_2"),
   ]
   layer_3_gradcam_images = [
      ("gradcam/gradcam_layer_3_1.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_2.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_3.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_4.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_5.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_6.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_7.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_8.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_9.png",  "layer_3"),
      ("gradcam/gradcam_layer_3_10.png", "layer_3"),
   ]
   layer_map = {
      "layer_1" : layer_1_gradcam_images,
      "layer_2" : layer_2_gradcam_images,
      "layer_3" : layer_3_gradcam_images,
   }
   gradcam_images = []
   num_of_layers = len(layers)
   for i in range(int(gradcam_number)):
      for layer in layers:
        gradcam_images.append(layer_map[layer][i])
      if num_of_layers == 2:

          gradcam_images.append(empty_image)
      if num_of_layers == 1:

          gradcam_images.append(empty_image)
          gradcam_images.append(empty_image)
          
      
   return  gradcam_images