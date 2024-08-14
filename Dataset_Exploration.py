import glob
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset



def plot_random(num = 4):

    tumor = [] 
    path = './Brain Tumor Data Set/Brain Tumor/*'
    for f in glob.iglob(path): # Loop through all the images in the folder
        img = cv2.imread(f) 
        img = cv2.resize(img, (128,128)) # Resize the image to 128x128
        b, g, r = cv2.split(img) # Split the image into its channels
        img = cv2.merge([r, g, b]) # Merge the channels
        tumor.append(img)

    healthy = [] 
    path = './Brain Tumor Data Set/Healthy/*'
    for f in glob.iglob(path): # Loop through all the images in the folder
        img = cv2.imread(f) 
        img = cv2.resize(img, (128,128)) # Resize the image to 128x128
        b, g, r = cv2.split(img) # Split the image into its channels
        img = cv2.merge([r, g, b]) # Merge the channels
        healthy.append(img)


    # Convert the list to a numpy array
    healthy = np.array(healthy) 
    tumor = np.array(tumor)
    healthy_imgs = healthy[np.random.choice(healthy.shape[0], num, replace = False)]
    tumor_imgs = tumor[np.random.choice(tumor.shape[0], num, replace = False)]

    plt.figure(figsize = (16, 9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title('Healthy')
        plt.imshow(healthy_imgs[i])
        

    plt.figure(figsize = (16, 9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title('tumor')
        plt.imshow(tumor_imgs[i]) 
        
