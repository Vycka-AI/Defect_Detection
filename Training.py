print("Importing libraries")
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from utils.dataloader import Duomenys
from utils.model import CustomVGG
print("Imported libraries")

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

def train(dataloader, model, optimizer, loss_fn, epochs, device, target_accuracy=None):
    """
    Script to train a model. Returns trained model.
    """
    model.to(device)
    #Set model to training mode
    model.train()

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}:", end=" ")
        running_loss = 0
        running_corrects = 0
        n_samples = 0

        #Užkraunamos nuotraukos ir pavadinimai
        for inputs, labels in dataloader:
            #Užkraunami duomenys į GPU arba CPU
            inputs = inputs.to(device)
            labels = labels.to(device)

            #Nunulinami gradientų parametrai
            optimizer.zero_grad()

            #Skaičiuojami spėjimai
            preds_scores = model(inputs)
            preds_class = torch.argmax(preds_scores, dim=-1)

            #Skaičiuojama paklaida
            loss = loss_fn(preds_scores, labels)

            #Skaičiuojamas gradienatas d_loss/d_x
            loss.backward()

            #Optimizer pakeičia parametrų vertes į priešingą pusę negu d_loss/d_x
            optimizer.step()

            #Skaičuojama suminė paklaida
            running_loss += loss.item() * inputs.size(0) #einamasis loss+=partijos loss*partijos dydis
            running_corrects += torch.sum(preds_class == labels)
            n_samples += inputs.size(0)

        epoch_loss = running_loss / n_samples #epochos (klaidos tikimybes) skaiciavimas
        epoch_acc = running_corrects.double() / n_samples #epochos accuracy (tikslumo) skaiciavimas)
        print("Loss = {:.4f}, Tikslumas = {:.4f}".format(epoch_loss, epoch_acc))

        if target_accuracy != None: #kai norimas tikslumas nustatytas
            if epoch_acc > target_accuracy: #nutraukti jei pasiektas norimas tikslumas
                print("Stabdoma...")
                break

    return model

if __name__ == "__main__":
    
    main_folder = "hazelnut"
    print(f"Data folder {main_folder}")

    target_train_accuracy = 0.98
    learning_rate = 0.0001
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mokymui naudojamas: {device}")
    #print(device)
    heatmap_thres = 0.7

    #Duomenų užkrovimui sukuriami du objektai
    train_loader, test_loader = Get_Train_test_loaders(main_folder, batch_size=10, test_size=0.2, random_state=42)

    #Sukonstruojamas modelis
    model = CustomVGG()
    #Geriems paveiksliukams duodamas mažesnis svoris, defektuotiems - didesnis
    class_weight = [1, 3]
    class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    #Parenkamas optimizatorius mokymui
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model = train(train_loader, model, optimizer, loss_fn, num_epochs, device, target_train_accuracy)

    model_path = "weights/Modeliukas_changed.h5"
    torch.save(model, model_path)
    # evaluate(model, test_loader, device)
    # model = torch.load(model_path, map_location=device)
