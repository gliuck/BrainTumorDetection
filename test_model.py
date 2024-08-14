import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import itertools


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Funzione per mostrare la matrice di confusione 
def show_confusion_matrix(cm, CLA_label, title='Confusion matrix', cmap=plt.cm.YlGnBu):
    
    plt.figure(figsize=(10,7))
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CLA_label))

    plt.xticks(tick_marks, [f"{value}={key}" for key , value in CLA_label.items()], rotation=45)
    plt.yticks(tick_marks, [f"{value}={key}" for key , value in CLA_label.items()])

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i,j]}\n{cm[i,j]/np.sum(cm)*100:.2f}%", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


def test_model(model, test_dataloader, seed=None):
    # Set seed for reproducibility
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval() # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():
        for D in test_dataloader:

            image = D['image'].to(device)
            label = D['label'].to(device)

            y_hat = model(image)
            predicted = torch.round(y_hat.squeeze())

            predictions.extend(predicted.tolist())
            true_labels.extend(label.tolist())

    cm = confusion_matrix(true_labels, predictions)

    test_accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)

    print(f'Test Accuracy: {test_accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    #plot confusion matrix
    show_confusion_matrix(cm, {0: 'Healthy', 1: 'Tumor'}, title='Confusion Matrix')

    #plot ROC curve
    # Binarizzazione delle etichette
    lb = LabelBinarizer()
    true_labels_bin = lb.fit_transform(true_labels)
    predictions_bin = lb.transform(predictions)

    # Calcolo della curva ROC e dell'AUC
    fpr, tpr, _ = roc_curve(true_labels_bin, predictions_bin)
    roc_auc = auc(fpr, tpr)

    # Tracciamento della curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Rate of True Positives')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()

    print('AUC: %.2f' % roc_auc)

    