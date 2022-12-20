import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.dataloader import Duomenys
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, class_names="auto"):
    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=[5, 5])
    sns.heatmap(
        confusion,
        annot=True,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.ylabel("Tikri")
    plt.xlabel("Spėti")
    plt.title("Spėjimo matrica")
    plt.show()

def Get_Train_test_loaders(main_folder, batch_size, test_size=0.2, random_state=42):

    #Užkraunami duomenys iš folderiu
    dataset = Duomenys(main_folder)

    #Paimami atsitiktiniai duomenys testavimui ir treniravimui su tam tikru duomenu santykiu
    train_idx, test_idx = train_test_split(
        np.arange(dataset.__len__()),
        test_size=test_size,
        shuffle=True,
        stratify=dataset.img_labels_detailed,
        random_state=random_state,
    )

    #Sukuriamas objketas duomenų surinkimui, kuris naudojamas su DataLoader
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    #Sukuriamas objketas duomenims, jame saugomos paveikslėlių matricos ir pavadinimai
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False)
    return train_loader, test_loader

def predict_and_localize(model, dataloader, device, thres=0.8, n_samples=9, show_heatmap=False):
    #atlieka spejima su duomenimis is dataloader
    #parodo img, tikra label, nuspejama label, tikimybe
    #jei nuspejamas defektas, jis apibreziamas
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
        out = model(inputs)
        probs, class_preds = torch.max(out[0], dim=-1)
        feature_maps = out[1].to("cpu")

        for img_i in range(inputs.size(0)):
            img = transform_to_PIL(inputs[img_i])
            class_pred = class_preds[img_i]
            prob = probs[img_i]
            label = labels[img_i]
            #Konvertuojama į numpy objektą
            heatmap = feature_maps[img_i][1].detach().numpy()

            counter += 1
            plt.subplot(n_rows, n_cols, counter)
            plt.imshow(img)
            plt.axis("off")
            
            plt.title(
                "Spėtas: {}, Tikimybė: {:.3f}, Tikras: {}".format(
                    class_names[class_pred], prob, class_names[label]
                )
                ,color="green" if class_names[class_pred] == class_names[label] else "red"
            )

            if class_pred == 1: #Rastas defektas
                if show_heatmap:
                    plt.imshow(heatmap, cmap="gist_gray", alpha=0.55)

            if counter == n_samples:
                plt.tight_layout()
                plt.show()
                return

if __name__ == "__main__":

    model_path = f"weights/Modeliukas_changed.h5"
    model = torch.load(model_path)

    train_loader, test_loader = Get_Train_test_loaders("hazelnut", batch_size=10, test_size=0.2, random_state=42)

    heatmap_thres = 0.5
    print(model)

    predict_and_localize(model, test_loader, "cpu", thres=heatmap_thres, n_samples=9, show_heatmap=True)