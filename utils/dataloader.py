import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class Duomenys(Dataset):
    def __init__(self, main_folder):
        #Įkęliamo paveikslėlio dydis

        INPUT_IMG_SIZE = (224, 224)

        #Klasifikuojamos klasės į geras ir defektuotas
        self.classes = ["Geras", "Blogas"]
        
        # --------------  Pakeičia paveikslėlio dydį ir konvertuoja į tensoriaus pavidalą vienu metu ------------- #
        #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        self.img_transform = transforms.Compose([transforms.Resize(INPUT_IMG_SIZE), transforms.ToTensor()])

        #Užkraunami treniruojami ir testuojami paveiksliukų aplankai
        (self.img_filenames, self.img_labels, self.img_labels_detailed) = self.get_images_and_labels(main_folder)

    def get_images_and_labels(self, main_folder):
        # --------------  Užkraunamos paveikslėlių failų vietos ir pavadinimai ----------------- #
        image_names = []
        labels = []
        labels_detailed = []
        for folder in ["train", "test"]:
            folder = os.path.join(main_folder, folder)

            for class_folder in os.listdir(folder):
                if class_folder == "good":
                    label = 0
                else:
                    label = 1

                label_detailed = class_folder

                #Užkraunamas aplankas, kuriame paveikslėliai
                class_folder = os.path.join(folder, class_folder)
                class_images = os.listdir(class_folder)

                class_img = []
                #Užkraunamos paveikslėlių nuorodos
                for image in class_images:
                    if image.find(".png") > -1:
                        class_img.append(os.path.join(class_folder, image))

                image_names.extend(class_img)
                labels.extend([label] * len(class_images))
                labels_detailed.extend([label_detailed] * len(class_images))

        print("Duomenu aplankas {}: N Paveiksliuku = {}".format(main_folder, len(labels)))
        return image_names, labels, labels_detailed

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #Naudojama iškviečiant paveiksliuką
        img_fn = self.img_filenames[idx]
        label = self.img_labels[idx]
        img = Image.open(img_fn)
        img = self.img_transform(img)
        label = torch.as_tensor(label, dtype=torch.long)
        return img, label