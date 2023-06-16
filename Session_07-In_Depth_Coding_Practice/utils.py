import matplotlib.pyplot as plt

def plot_images(num_of_images:int, images):
  '''
  Plot the input images
  '''
  figure = plt.figure()
  for index in range(1, num_of_images + 1):
    plt.subplot(int(num_of_images/10), 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze())
