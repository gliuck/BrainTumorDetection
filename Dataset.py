import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt

#create a dataset class that extends the Dataset class from PyTorch

class CustomBrainDataset(Dataset):
    # The principal function of the class are the __init__ and __len__  and __getitem__  functions
    # The __init__ function is used to initialize the dataset

    def __init__(self, mode='train'):

        # Variable to hold the Training data and Test data
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = None, None, None, None, None, None
        # A variable to determine if we interested in retrieving the training OR the validation data
        # self.mode ='train'
        self.mode = mode
        
        healthy = [] 
        tumor = [] 
    
        path = './Brain Tumor Data Set/Brain Tumor/*'
        for f in glob.iglob(path): # Loop through all the images in the folder
            img = cv2.imread(f)
            img = cv2.resize(img, (128,128)) # Resize the image to 128x128
            b, g, r = cv2.split(img) # Split the image into its channels
            img = cv2.merge([r, g, b]) # Merge the channels
            img = img.reshape((img.shape[2], img.shape[0], img.shape[1])) # Change the shape of the image to C x H x W
            tumor.append(img)

        
        path = './Brain Tumor Data Set/Healthy/*'
        for f in glob.iglob(path): # Loop through all the images in the folder
            img = cv2.imread(f) 
            img = cv2.resize(img, (128,128)) # Resize the image to 128x128
            b, g, r = cv2.split(img) # Split the image into its channels
            img = cv2.merge([r, g, b]) # Merge the channels
            img = img.reshape((img.shape[2], img.shape[0], img.shape[1])) # Change the shape of the image to C x H x W
            healthy.append(img)

        # our images
        # Convert the list to a numpy array
        healthy = np.array(healthy, dtype=np.float32)
        tumor = np.array(tumor, dtype=np.float32) 

        # our labels
        tumor_label = np.ones( tumor.shape[0], dtype=np.float32 ) # Create an array of ones with the same length as the tumor array
        healthy_label = np.zeros( healthy.shape[0], dtype=np.float32 )

        # Concatenate the two arrays, healthy and tumor
        self.images = np.concatenate((tumor, healthy), axis=0)  
        self.labels = np.concatenate((tumor_label, healthy_label))
        
        # Split data into train and test
        self.train_val_split()
        
        # Normalize the images
        self.images = self.images / 255.0

        # # Definisci le trasformazioni necessarie per convertire le immagini in tensori
        # transform = transforms.Compose([
        #     transforms.ToTensor(),  # Converte l'immagine in un tensor
        #     # Potresti aggiungere altre trasformazioni qui, come resize, normalizzazione, ecc.
        # ])

    # Split the data into trainin, val and test sets 
    def train_val_split(self):
        self.X_train, self.X_val, self.y_train, self.y_val = \
        train_test_split(self.images, self.labels, test_size=0.3, stratify=self.labels, shuffle=True, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = \
        train_test_split(self.X_val, self.y_val, test_size=0.4, shuffle=True, random_state=42) 

    # The __len__ function returns the length of the dataset
    def __len__(self):

        if self.mode == 'train':
            return len(self.X_train)
        elif self.mode == 'val':
            return len(self.X_val)
        elif self.mode == 'test':
            return len(self.X_test) 


    def __getitem__(self, index):
        if self.mode == 'train':
            sample = {'image': self.X_train[index], 'label': self.y_train[index]}
        elif self.mode == 'val':
            sample = {'image': self.X_val[index], 'label': self.y_val[index]}
        elif self.mode == 'test':
            sample = {'image': self.X_test[index], 'label': self.y_test[index]}
        return sample


    def plot_random(self, num=5, mode='train'):
    
        if mode == 'train':
            images = self.X_train
            labels = self.y_train
        elif mode == 'val':
            images = self.X_val
            labels = self.y_val
        elif mode == 'test':
            images = self.X_test
            labels = self.y_test
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'val', 'test'.")

        # Filtra le immagini in base alle etichette
        healthy_indices = np.where(labels == 0)[0]  # Indici delle immagini sane
        tumor_indices = np.where(labels == 1)[0]  # Indici delle immagini con tumore

        # Seleziona casualmente gli indici per le immagini sane e con tumore
        healthy_sample_indices = np.random.choice(healthy_indices, num, replace=False)
        tumor_sample_indices = np.random.choice(tumor_indices, num, replace=False)

        # Seleziona le immagini corrispondenti agli indici
        healthy_imgs = images[healthy_sample_indices]
        tumor_imgs = images[tumor_sample_indices]

        # Visualizza le immagini
        plt.figure(figsize=(16, 9))
        for i, img in enumerate(healthy_imgs):
            plt.subplot(2, num, i+1)
            plt.title('Healthy')
            img = np.transpose(img, (1, 2, 0))  # Riorganizza i canali all'ultimo indice
            plt.imshow(img, cmap='gray')

        plt.figure(figsize=(16, 9))
        for i, img in enumerate(tumor_imgs):
            plt.subplot(2, num, i+1)
            plt.title('Tumor')
            img = np.transpose(img, (1, 2, 0))  # Riorganizza i canali all'ultimo indice
            plt.imshow(img, cmap='gray')