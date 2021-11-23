import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from lib.resnet_hierarchy import resnet50, resnet101

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False,
                    use_pretrained=True, logger=None):
    model_ft = None

    if model_name == "resnet101":
        """ Resnet101
        """
        # model_ft = models.resnet101(pretrained=use_pretrained)
        model_ft = resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet50":
        """ Resnet50
        """
        # model_ft = models.resnet50(pretrained=use_pretrained)
        model_ft = resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        W_s2g = np.load('data/semi_inat/taxa_weights.npz')['W_s2g']
        W_g2f = np.load('data/semi_inat/taxa_weights.npz')['W_g2f']
        W_f2o = np.load('data/semi_inat/taxa_weights.npz')['W_f2o']
        W_o2c = np.load('data/semi_inat/taxa_weights.npz')['W_o2c']
        W_c2p = np.load('data/semi_inat/taxa_weights.npz')['W_c2p']
        W_p2k = np.load('data/semi_inat/taxa_weights.npz')['W_p2k']
        model_ft.W_s2g = torch.tensor(W_s2g, requires_grad=False)
        model_ft.W_g2f = torch.tensor(W_g2f, requires_grad=False)
        model_ft.W_f2o = torch.tensor(W_f2o, requires_grad=False)
        model_ft.W_o2c = torch.tensor(W_o2c, requires_grad=False)
        model_ft.W_c2p = torch.tensor(W_c2p, requires_grad=False)
        model_ft.W_p2k = torch.tensor(W_p2k, requires_grad=False)
        model_ft.W_s2g = model_ft.W_s2g.float().to(device)
        model_ft.W_g2f = model_ft.W_g2f.float().to(device)
        model_ft.W_f2o = model_ft.W_f2o.float().to(device)
        model_ft.W_o2c = model_ft.W_o2c.float().to(device)
        model_ft.W_c2p = model_ft.W_c2p.float().to(device)
        model_ft.W_p2k = model_ft.W_p2k.float().to(device)


    return model_ft