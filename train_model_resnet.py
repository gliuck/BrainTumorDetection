import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


def train_model_resnet(model, train_dataloader, val_dataloader, optimizer, criterion, num_epochs, patience, seed=None):
    # Set seed for reproducibility
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # resnet18.to(device)
    model.to(device)

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Train loop
        model.train()
        for batch in train_dataloader:
            inputs, labels = batch['image'].to(device), batch['label'].to(device).long()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate train loss and accuracy
        train_loss /= len(train_dataloader.dataset)
        train_accuracy = 100. * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # val loop
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                images, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.long())
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate val loss and accuracy
        val_loss /= len(val_dataloader.dataset)
        val_accuracy = 100. * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.2f}%,  Val Acc: {val_accuracy:.2f}%')

        # Early stopping
        if epoch > patience:
            if val_losses[-patience] < min(val_losses):
                print(f'Early stopping at epoch {epoch+1}')
                break


    # Plotting
    plt.figure(figsize=(10, 5))

    # Plot training and val loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    # Plot training and val accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Train')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

