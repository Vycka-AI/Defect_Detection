import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from utils.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utils.model import CustomVGG
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS
import torch.nn.functional as F

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def predict_localize(
    model, dataloader, device, thres=0.8, n_samples=9, show_heatmap=False
):
    """
    Runs predictions for the samples in the dataloader.
    Shows image, its true label, predicted label and probability.
    If an anomaly is predicted, draws bbox around defected region and heatmap.
    """
    model.to(device)
    model.eval()

    class_names = dataloader.dataset.classes
    transform_to_PIL = transforms.ToPILImage()

    n_cols = 3
    n_rows = int(np.ceil(n_samples / n_cols))
    plt.figure(figsize=[n_cols * 5, n_rows * 5])

    counter = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        model.feature_extractor.register_forward_hook(get_activation('feature_extractor'))
        out = model(inputs)
        feature_maps = out[1].to("cpu")
        #plt.imshow((torch.reshape(activation['feature_extractor'], (1024, 980))))
        """
        probs, class_preds = torch.max(out[0], dim=-1)
        feature_maps = out[1].to("cpu")

        for img_i in range(inputs.size(0)):
            img = transform_to_PIL(inputs[img_i])
            class_pred = class_preds[img_i]
            prob = probs[img_i]
            label = labels[img_i]
            heatmap = feature_maps[img_i][NEG_CLASS].detach().numpy()

            counter += 1
            plt.subplot(n_rows, n_cols, counter)
            plt.imshow(img)
            plt.axis("off")
            
            plt.title(
                "Predicted: {}, Prob: {:.3f}, True Label: {}".format(
                    class_names[class_pred], prob, class_names[label]
                )
                ,color="green" if class_names[class_pred] == class_names[label] else "red"
            )

            if class_pred == NEG_CLASS:
                if show_heatmap:
                    plt.imshow(heatmap, cmap="inferno", alpha=0.3)

            if counter == n_samples:
                plt.tight_layout()
                plt.show()
                return
            """

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model_path = f"weights/Modeliukas_changed.h5"
model = torch.load(model_path)

train_loader, test_loader = get_train_test_loaders(
    root="hazelnut", batch_size=10, test_size=0.2, random_state=42,
)

heatmap_thres = 0.5
print(model)
#model.fc2.register_forward_hook(get_activation('fc2'))


predict_localize(
    model, test_loader, "cpu", thres=heatmap_thres, n_samples=9, show_heatmap=True
)