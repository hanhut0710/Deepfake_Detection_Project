import torchvision.models as models
import torch.nn as nn
import torch
from config import Config

def get_model(model_name="resnet18", num_classes=1):
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name. Use 'resnet18' or 'efficientnet_b0' or 'mobilenet_v3'.")

    return model.to(Config.device)
   