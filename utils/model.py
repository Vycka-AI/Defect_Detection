import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

INPUT_IMG_SIZE = (224, 224)
class CustomVGG(nn.Module):
    #Naudojamas feature extractor VGG16, jau apmokytas modelis, kad nereiktų naudoti labai daug duomenų
    def __init__(self):
        super(CustomVGG, self).__init__()
        self.feature_extractor = models.vgg16(weights='VGG16_Weights.DEFAULT').features[:-1]
        self.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2) #Max pooling, kernel_size - branduolys, stride - kiek branduolys pasislenka po kiekvieno skaičiavimo
        self.AvgPool2d = nn.AvgPool2d(kernel_size=(7, 7)) #Average pooling
        #Sudaromas 1 - dimensinis vektorius iš tensoriaus
        self.Flatten = nn.Flatten()
        #in_features = įėjimų skaičius, out_features = išėjimų skaičius
        self.Linear = nn.Linear(in_features=512, out_features=2)
        #vgg16 Convolutional pirmi 23 sluoksniai, padaromi, kad jų neįtakotų treniravimas
        for param in self.feature_extractor[:23].parameters():
            param.requires_grad = False

    def forward(self, x):
        # ------------ Paveikslėlio perdavimas nauroniniam tinklui ------------------ #

        # ------------ Perduodama feature extractor ------------------ #
        feature_maps = self.feature_extractor(x)

        # ------------ Perduodama sukonstruotam tinklui -------------- #
        x = self.MaxPool2d(feature_maps)
        x = self.AvgPool2d(x)        
        x = self.Flatten(x)
        scores = self.Linear(x)

        # scores - 512 verčių, kurios turi būti paverčiamos į žmogui skaitomą pavidalą ---------- #
        # softmax konvertuoja vektorių iš K skaičių į tikimybės pasiskirstymą į K galimų reikšmių [0,1]
        if self.training:
            return scores

        else:
            # ----------- Galutinės vertės, normalizuojamas tensorius    ---------- #
            probs = nn.functional.softmax(scores, dim=-1)
            # ----------- Galutiniai svoriai  ---------- #
            weights = self.Linear.weight
            # Panaikinamos tensoriaus 1 dimensinės reikšmės
            print(weights)
            weights = (weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).repeat((x.size(0), 1, 1, 14, 14)))
            print(weights)
            feature_maps = feature_maps.unsqueeze(1).repeat((1, probs.size(1), 1, 1, 1))
            location = torch.mul(weights, feature_maps).sum(axis=2) #Dauginama iš svorių, sumuojama, kad sumažėtų dimensijų skaičius
            location = F.interpolate(location, size=INPUT_IMG_SIZE, mode="bilinear") #Panaudojama, kad iš 14 x 14 Heatmap pavirstų į turimo paveikslėlio dydį atvaizduoti defektams

            #Normavimas heatmap
            maxs, _ = location.max(dim=-1, keepdim=True)
            maxs, _ = maxs.max(dim=-2, keepdim=True)
            mins, _ = location.min(dim=-1, keepdim=True)
            mins, _ = mins.min(dim=-2, keepdim=True)
            norm_location = (location - mins) / (maxs - mins)

            return probs, norm_location