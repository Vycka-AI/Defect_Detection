import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS

print("Imprted libraries")
data_folder = "hazelnut"
print(f"Data folder {data_folder}")

batch_size = 10
target_train_accuracy = 0.98
lr = 0.0001
epochs = 10
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Mokymui naudojamas: {device}")
#print(device)
heatmap_thres = 0.7
n_cv_folds = 5

train_loader, test_loader = get_train_test_loaders(
    root=data_folder, batch_size=batch_size, test_size=0.2, random_state=42,
)


model = CustomVGG()

class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=lr)

model = train(
    train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy
)

model_path = "weights/Modeliukas.h5"
torch.save(model, model_path)
# model = torch.load(model_path, map_location=device)




def train(
    dataloader, model, optimizer, criterion, epochs, device, target_accuracy=None
):
    """
    Script to train a model. Returns trained model.
    """
    model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}:", end=" ")
        running_loss = 0
        running_corrects = 0
        n_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds_scores = model(inputs)
            preds_class = torch.argmax(preds_scores, dim=-1)
            loss = criterion(preds_scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds_class == labels)
            n_samples += inputs.size(0)

        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples
        print("Loss = {:.4f}, Accuracy = {:.4f}".format(epoch_loss, epoch_acc))

        if target_accuracy != None:
            if epoch_acc > target_accuracy:
                print("Early Stopping")
                break

    return model