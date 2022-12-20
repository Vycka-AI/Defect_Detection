print("Importing libraries...")
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
print("Imported...")

from utils.dataloader import get_train_test_loaders
from utils.helper import train, evaluate, predict_localize
from utils.constants import NEG_CLASS

GOOD_CLASS_FOLDER = "good"
DATASET_SETS = ["train", "test"]
IMG_FORMAT = ".png"
INPUT_IMG_SIZE = (224, 224)
NEG_CLASS = 1

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

class MVTEC_AD_DATASET(Dataset):
    """
    Class to load subsets of MVTEC ANOMALY DETECTION DATASET
    Dataset Link: https://www.mvtec.com/company/research/datasets/mvtec-ad
    
    Root is path to the subset, for instance, `mvtec_anomaly_detection/leather`
    """

    def __init__(self, root):
        if NEG_CLASS == 1:
            self.classes = ["Good", "Anomaly"]
        else:
            self.classes = ["Anomaly", "Good"]
        self.img_transform = transforms.Compose([transforms.Resize(INPUT_IMG_SIZE), transforms.ToTensor()])
        (self.img_filenames, self.img_labels, self.img_labels_detailed,) = self._get_images_and_labels(root)

    def _get_images_and_labels(self, root):
        image_names = []
        labels = []
        labels_detailed = []

        for folder in DATASET_SETS:
            folder = os.path.join(root, folder)
            print(f"Found folders: {os.listdir(folder)}")
            for class_folder in os.listdir(folder):
                if class_folder == GOOD_CLASS_FOLDER:
                    label = 1 - NEG_CLASS
                else:
                    label = NEG_CLASS
                label_detailed = class_folder
                class_folder = os.path.join(folder, class_folder)
                class_images = os.listdir(class_folder)
                class_images = [
                    os.path.join(class_folder, image)
                    for image in class_images
                    if image.find(IMG_FORMAT) > -1
                ]
                #print(folder)
                image_names.extend(class_images)
                labels.extend([label] * len(class_images))
                labels_detailed.extend([label_detailed] * len(class_images))

        print(f"Dataset {root}: N Images = {len(labels)}, Share of anomalies = {np.sum(labels) / len(labels):.3f}")
        return image_names, labels, labels_detailed

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_fn = self.img_filenames[idx]
        label = self.img_labels[idx]
        img = Image.open(img_fn)
        img = self.img_transform(img)
        label = torch.as_tensor(label, dtype=torch.long)
        return img, label


def get_cv_train_test_loaders(root, batch_size, n_folds=5):
    """
    Returns train and test dataloaders for N-Fold cross-validation.
    Splits dataset in stratified manner, considering various defect types.
    """
    dataset = MVTEC_AD_DATASET(root=root)

    kf = StratifiedKFold(n_splits=n_folds)
    kf_loader = []

    for train_idx, test_idx in kf.split(np.arange(dataset.__len__()), dataset.img_labels_detailed):
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False)
        kf_loader.append((train_loader, test_loader))
    return kf_loader

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
                weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
                .repeat((x.size(0), 1, 1, INPUT_IMG_SIZE[0] // 2 ** 4, INPUT_IMG_SIZE[0] // 2 ** 4,)))
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
    print(cv_folds)
    class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    for i, (train_loader, test_loader) in enumerate(cv_folds):
        print(f"Fold {i+1}/{n_cv_folds}")
        #model = CustomVGG(input_size)
        model = CustomVGG(2)
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model = train(train_loader, model, optimizer, criterion, epochs, device)
        evaluate(model, test_loader, device)
    model_path = "weights/Modeliukas_CrossValid.h5"
    torch.save(model, model_path)