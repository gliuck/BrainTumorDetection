# Brain Tumor Detection 🧠

## Introduction 📝

This project focuses on detecting brain tumors using deep learning techniques. The goal is to classify brain MRI images as either showing a tumor or not. Multiple convolutional neural network (CNN) architectures were implemented, including a custom CNN, a pre-trained ResNet with fine-tuning, and an improved CNN model.

## Task 🎯

The primary objective is to build and evaluate deep learning models that can accurately classify brain MRI images into two categories: tumor and non-tumor. This involves dataset exploration, model development, and performance comparison.

## Dataset 📁

The data used in this project consists of a dataset of brain magnetic resonance imaging (MRI) images from patients with brain tumors. The dataset has been divided into two classes: images with tumor presence and normal images. In total, the dataset contains 4,600 images, with 2,513 brain tumor images and 2,087 normal images.
Key steps involved in dataset handling include:

- **Loading the Dataset**: Importing and examining the dataset.
- **Reading the Images**: Processing and visualizing the MRI images.
- **Splitting the Dataset**: Dividing the dataset into training, validation, and test sets.
- **Creating Data Loaders**: Setting up data loaders for efficient model training and evaluation.
#### Below are two sample images from the data set:
| MRI Scan without a Tumor | MRI Scan with a Tumor |
|:-------------------------:|:----------------------:|
| <img src="https://github.com/user-attachments/assets/ca155d90-b56c-49f1-8c23-6dca4c96517f" alt="No Tumor" width="300"/> | <img src="https://github.com/user-attachments/assets/a38f4a4d-bb67-4493-a2ff-ddcf28e93020" alt="Yes Tumor" width="300"/> |

## Libraries Used 📚

- `torch`
- `torchvision`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `torchsummary`
- `torchviz`

## Models Used 🧠

### 1. Custom CNN

- **Architecture**: A custom CNN was designed specifically for this task. It includes three convolutional blocks followed by a fully connected block:
  - **Convolutional Blocks**: 
    - Block 1: Conv2d (3x32), BatchNorm2d, ReLU, MaxPool2d.
    - Block 2: Conv2d (32x64), BatchNorm2d, ReLU, MaxPool2d.
    - Block 3: Conv2d (64x128), BatchNorm2d, ReLU, MaxPool2d.

  - **Fully Connected Block**: 
    - Linear(128x16x16 -> 512), ReLU, Dropout(0.5).
    - Linear(512 -> 256), ReLU, Dropout(0.5).
    - Linear(256 -> 120), ReLU, Dropout(0.5).
    - Linear(120 -> 84), ReLU, Dropout(0.5).
    - Linear(84 -> 1), Sigmoid.
    - ```plaintext
       INPUT → [CONV → BATCHNORM → RELU → MAXPOOL] × 3 → [FC → RELU → DROPOUT] × 4 → FC → SIGMOID → OUTPUT
 <img src="https://github.com/user-attachments/assets/d8a3adb2-eecd-4da5-af49-d01c3d85bdef" alt="cnn1" width="700"/>



- **Training**:
  - **Loss Function**: Binary Cross-Entropy Loss.
  - **Optimizer**: Adam optimizer with a learning rate of 0.001.
  - **Epochs**: The model was trained for multiple epochs with early stopping to prevent overfitting.
  - **Evaluation**: The model was evaluated using accuracy, precision, recall, F1 score, and confusion matrix.

### 2. Pre-Trained ResNet with Fine-Tuning

- **Architecture**: A ResNet-18 model pre-trained on ImageNet was fine-tuned for the binary classification task.
  - The final fully connected layer was replaced with a custom layer to output a single value followed by a Sigmoid activation function.

- **Training**:
  - **Loss Function**: Binary Cross-Entropy Loss.
  - **Optimizer**: Adam optimizer with a learning rate of 0.001.
  - **Epochs**: The model underwent fine-tuning for several epochs with early stopping based on validation performance to avoid overfitting.
  - **Evaluation**: Accuracy, precision, recall, F1 score, and confusion matrix were used to assess performance.

### 3. Improved CNN

- **Architecture**: An enhanced CNN architecture was developed, featuring additional convolutional layers and dropout for improved regularization.
  - The architecture was similar to the custom CNN, but with architectural improvements for better generalization.
  - ```plaintext
    INPUT → [CONV → BATCHNORM → RELU → MAXPOOL] × 4 → [FC → RELU → DROPOUT] × 2 → FC → SIGMOID → OUTPUT
- **Training**:
  - **Loss Function**: Binary Cross-Entropy Loss.
  - **Optimizer**: Adam optimizer with a learning rate of 0.001.
  - **Epochs**: The model was trained with architectural improvements to boost performance.
  - **Evaluation**: The model was evaluated using accuracy, precision, recall, F1 score, and confusion matrix.

## Experiments 🔬

### Development Environment

The project was developed on a Windows 11 system using Python, with libraries including PyTorch, NumPy, Pandas, Matplotlib, Scikit-learn, and Jupyter Notebook. The hardware used was an AMD Ryzen 5 processor with 16GB RAM and an NVIDIA GeForce RTX 3060 GPU.

### Model Training

#### Custom CNN Training

The custom CNN was trained with the following parameters:

- **Epochs**: 50
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Dropout Rate**: 0.6
- **Optimizer**: Adam
- **Loss Function**: BCELoss
- **Patience**: 5 (early stopping)

The network achieved an accuracy of 90.58% on the validation set.

#### ResNet18 Training

The ResNet18 model was fine-tuned for the classification task, achieving an accuracy of 96.01% on the validation set.

#### ImprovedCNN Training

The ImprovedCNN model, with a deeper architecture, was trained under similar conditions as the custom CNN. It achieved an accuracy of 93.96% on the validation set.

### Results

The performance of each model was evaluated on a separate test set:

| **Model**         | **Accuracy (%)** |
|-------------------|------------------|
| Custom CNN        | 93.65            |
| ResNet18          | 96.92            |
| ImprovedCNN       | 94.38            |

The ResNet18 model demonstrated superior performance, with the highest accuracy and generalization capabilities.

## Conclusion ✅

The models performed differently, with the improved CNN showing promising results for better generalization. Future work could involve further tuning of hyperparameters, increasing the dataset size, and experimenting with more complex models.

## Credits 🙏

The dataset used in this project was obtained from a publicly available source on brain tumor MRI images [Kaggle](https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset)
