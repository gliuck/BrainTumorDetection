import torch
import torch.nn as nn

# Define the model extending the nn.Module class from PyTorch
# Define Architecture for BrainCNN Model

class BrainCNN(nn.Module):
    def __init__(self):
        super(BrainCNN, self).__init__()
      
     # Blocco convoluzionale 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Blocco convoluzionale 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Blocco convoluzionale 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Calculating the input size to the fully connected block after convolutional blocks
        self.fc_input_size = 128 * 16 * 16

        # Blocco completamente connesso
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 120),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(84, 1),
            nn.Sigmoid()  # Funzione di attivazione per garantire che l'output sia compreso tra 0 e 1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1) 
        x = self.fc(x)
        return x

