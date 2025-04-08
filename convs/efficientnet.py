import torch
import torch.nn as nn
from torchvision import models

def efficientnet(pretrained=False, num_classes=1000, args=None):
    model = models.efficientnet_b0(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

class ModifiedEfficientNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, args=None):
        super(ModifiedEfficientNet, self).__init__()
        assert args is not None, "You should pass args to the model"

        self.args = args
        self.model = efficientnet(pretrained=pretrained, num_classes=num_classes, args=args)
        self.features = self.model.features

        # Define output feature dimension based on EfficientNet
        self.out_dim = self.model.classifier[1].in_features  # Typically 1280

        # Replace first conv layer for smaller image sizes like CIFAR
        if 'cifar' in args.get("dataset", "") or 'covid' in args.get("dataset", ""):
            self.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fmaps = []

        # Extract feature maps from key layers for interpretability
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in [2, 4, 6, 8]:  # Tuned indices for EfficientNet-B0 stages
                fmaps.append(x)

        # Global Average Pooling + Flatten
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)  # Shape: [B, 1280]

        return {
            'fmaps': fmaps,
            'features': x,
        }
