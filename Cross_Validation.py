import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS


class CustomVGG(nn.Module):
    """
    Custom multi-class classification model 
    with VGG16 feature extractor, pretrained on ImageNet
    and custom classification head.
    Parameters for the first convolutional blocks are freezed.
    
    Returns class scores when in train mode.
    Returns class probs and normalized feature maps when in eval mode.
    """

    def __init__(self, n_classes=2):
        super(CustomVGG, self).__init__()
        self.feature_extractor = models.vgg16(pretrained=True).features[:-1]
        self.classification_head = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=(INPUT_IMG_SIZE[0] // 2 ** 5, INPUT_IMG_SIZE[1] // 2 ** 5)),
            nn.Flatten(),
            nn.Linear(in_features=self.feature_extractor[-2].out_channels,out_features=n_classes,),
            )
        self._freeze_params()

    def _freeze_params(self):
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False

    def forward(self, x):
        feature_maps = self.feature_extractor(x)
        scores = self.classification_head(feature_maps)

        if self.training:
            return scores

        else:
            probs = nn.functional.softmax(scores, dim=-1)

            weights = self.classification_head[3].weight
            weights = (
                weights.unsqueeze(-1)
                .unsqueeze(-1)
                .unsqueeze(0)
                .repeat(
                    (
                        x.size(0),
                        1,
                        1,
                        INPUT_IMG_SIZE[0] // 2 ** 4,
                        INPUT_IMG_SIZE[0] // 2 ** 4,
                    )
                )
            )
            feature_maps = feature_maps.unsqueeze(1).repeat((1, probs.size(1), 1, 1, 1))
            location = torch.mul(weights, feature_maps).sum(axis=2)
            location = F.interpolate(location, size=INPUT_IMG_SIZE, mode="bilinear")

            maxs, _ = location.max(dim=-1, keepdim=True)
            maxs, _ = maxs.max(dim=-2, keepdim=True)
            mins, _ = location.min(dim=-1, keepdim=True)
            mins, _ = mins.min(dim=-2, keepdim=True)
            norm_location = (location - mins) / (maxs - mins)

            return probs, norm_location

if __name__ == "__main__":

    data_folder = "hazelnut"
    batch_size = 10
    target_train_accuracy = 0.98
    lr = 0.0001
    class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    heatmap_thres = 0.7
    n_cv_folds = 5
    epochs = 10

    cv_folds = get_cv_train_test_loaders(
        root=data_folder,
        batch_size=batch_size,
        n_folds=n_cv_folds,
    )

    class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    for i, (train_loader, test_loader) in enumerate(cv_folds):
        print(f"Fold {i+1}/{n_cv_folds}")
        #model = CustomVGG(input_size)
        model = CustomVGG(2)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model = train(train_loader, model, optimizer, criterion, epochs, device)
        evaluate(model, test_loader, device)
    model_path = "weights/Modeliukas_CrossValid.h5"
    torch.save(model, model_path)