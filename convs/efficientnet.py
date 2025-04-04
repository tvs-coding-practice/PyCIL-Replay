import torch
import torch.nn as nn
from torchvision import models

def efficientnet(pretrained=False, num_classes=1000, args=None):
    model = models.efficientnet_b0(pretrained=pretrained)  # EfficientNet-B0 from torchvision
    
    # Modify the classifier to match your dataset's classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model


class ModifiedEfficientNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, args=None):
        super(ModifiedEfficientNet, self).__init__()
        assert args is not None, "You should pass args to the model"

        self.model = efficientnet(pretrained=pretrained, num_classes=num_classes, args=args)
        self.features = self.model.features

        self.out_dim = self.model.classifier[1].in_features

        # Adjust the first convolutional layer for datasets like CIFAR (smaller images)
        if 'cifar' in args.get("dataset", "") or 'covid' in args.get("dataset", ""):
            self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        # Initialize model weights
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
        # Feature map extraction (optional)
        x = self.model.features(x)
        fmaps.append(x)
        x = self.model.classifier[0](x)
        features = torch.flatten(x, 1)
        return {
            'fmaps': fmaps,
            'features': features,
        }
