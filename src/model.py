import torchvision.models as models
import torch.nn as nn
import torch

def get_model(num_classes=1, config=None):
    if config.MODEL_NAME == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config.MODEL_NAME == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif config.MODEL_NAME == "mobilenet_v3":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unsupported model name. Use 'resnet18' or 'efficientnet_b0' or 'mobilenet_v3'.")

    return model.to(config.device)