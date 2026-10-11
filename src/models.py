import torch.nn as nn
from torchvision import models

def build_model(name, num_classes=2, pretrained=True):
    name = name.lower()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        m = models.resnet18(weights=weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        m = models.densenet121(weights=weights)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
        return m
    if name in ("efficientnet_b0", "efficientnet-b0"):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        m = models.efficientnet_b0(weights=weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    raise ValueError(f"Unknown model: {name}")

def gradcam_target_layer(model, name):
    name = name.lower()
    if name == "resnet18":
        return model.layer4[-1].conv2
    if name == "densenet121":
        return model.features.denseblock4
    if name in ("efficientnet_b0", "efficientnet-b0"):
        return model.features[-1]
    raise ValueError(name)
