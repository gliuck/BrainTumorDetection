import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Keep track of epoch losses
epoch_train_loss = []
epoch_train_accuracy = []
epoch_val_loss = []
epoch_val_accuracy = []

def train_model_improved(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, patience, seed=None):
    # Set seed for reproducibility
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    for epoch in range(num_epochs):
        
        train_losses = []
        train_correct = 0
        train_total = 0
        
        model.train() # Set the model to training mode 
        # cbd.mode = 'train' # Set the mode to train in the dataset
        
        # The input is a tupla containing the image and the label
        for D in train_dataloader:
            
            optimizer.zero_grad()

            # Move the data to the device
            image = D['image'].to(device)
            label = D['label'].to(device)

            # Forward pass
            y_hat = model(image)
            # Calculate the loss
            loss = criterion(y_hat.squeeze(), label)

            # Back-propagation
            loss.backward() 
            optimizer.step()
            train_losses.append(loss.item())

            # Calculate accuracy
            predicted = torch.round(y_hat.squeeze())  # Round the output to 0 or 1
            train_correct += (predicted == label).sum().item()
            train_total += label.size(0)

        epoch_train_loss.append(np.mean(train_losses)) 
        epoch_train_accuracy.append(100.*(train_correct / train_total))
        

        # Test the model
        model.eval() # Set the model to evaluation mode
        # cbd.mode = 'val' # Set the mode to test
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for D in val_dataloader:
                image = D['image'].to(device)
                label = D['label'].to(device)

                y_hat = model(image)
                #error = nn.BCELoss()
                #loss = torch.sum(error(y_hat.squeeze(), label))
                loss = criterion(y_hat.squeeze(), label)
                val_losses.append(loss.item())

                #Calculate accuracy
                predicted = torch.round(y_hat.squeeze())  
                val_correct += (predicted == label).sum().item()
                val_total += label.size(0)

        epoch_val_loss.append(np.mean(val_losses))
        epoch_val_accuracy.append(100.*(val_correct / val_total))
        

        if (epoch+1) % 10 == 0:
            #print('Train Epoch : {} \tTrain Loss: {:.6f}\tTest Loss: {:.6f}\tTrain Accuracy: {:.4f}%\tTest Accuracy: {:.4f}%'.format(epoch+1, np.mean(train_losses), np.mean(val_losses), epoch_train_accuracy[-1], epoch_val_accuracy[-1]))
            print(f'Train Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(train_losses):.6f}, Val Loss: {np.mean(val_losses):.6f}, Train Accuracy: {epoch_train_accuracy[-1]:.2f}%, Val Accuracy: {epoch_val_accuracy[-1]:.2f}%')

        #apply early stopping

        if epoch > patience:
            if epoch_val_loss[-patience] < min(epoch_val_loss):
                print(f'Early stopping at epoch {epoch+1}')
                break

     
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(epoch_train_loss)+1), epoch_train_loss, label='Train')
    plt.plot(range(1, len(epoch_val_loss)+1), epoch_val_loss, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)


    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epoch_train_accuracy)+1), epoch_train_accuracy, label='Train')
    plt.plot(range(1, len(epoch_val_accuracy)+1), epoch_val_accuracy, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Add grid
    plt.grid(True)
    plt.tight_layout()
    plt.show()
