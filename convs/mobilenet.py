import torch
import torch.nn as nn
from torchvision import models
try:
    from torchvision.models.utils import load_state_dict_from_url
except:
    from torch.hub import load_state_dict_from_url

# MobileNetV2 Model URL (for pretrained weights)
model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
}

def mobilenetv2(pretrained=False, progress=True, **kwargs):
    """MobileNetV2 model from "MobileNetV2: Inverted Residuals and Linear Bottlenecks".
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    model = models.mobilenet_v2(pretrained=False, progress=progress, **kwargs)
    
    # Load the pretrained weights if necessary
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
        model.load_state_dict(state_dict)

    return model

class ModifiedMobileNetV2(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, args=None, **kwargs):
        super(ModifiedMobileNetV2, self).__init__()

        assert args is not None, "You should pass args to the model"

        # Load the pre-trained or fresh MobileNetV2 model
        self.model = mobilenetv2(pretrained=pretrained, num_classes=num_classes, **kwargs)
        
        # Feature map extraction (using layers of MobileNetV2)
        self.features = self.model.features

        # Adjust the classifier layer based on your input size
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # MobileNetV2 often has a dropout layer here
            nn.Linear(1280, num_classes)  # Change 5120 to 1280 (default for MobileNetV2)
        )
        
        # Output dimension (out_dim)
        self.out_dim = 1280  # Use 1280 instead of 5120

        # Adjust the first convolution layer based on the dataset
        if 'cifar' in args.get("dataset", "") or 'covid' in args.get("dataset", ""):
            self.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
      
        # Initialize the model weights (if necessary)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fmaps = []

        # Pass input through feature extractor
        for idx, layer in enumerate(self.features):
            x = layer(x)
            # print(f"Layer {idx} output shape: {x.shape}")  # 🔍 Debug print

            if idx in [4, 6, 9, 13]:  
                fmaps.append(x)

        # 🔧 Ensure GAP is applied before flattening
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1)  # (batch_size, 1280, 1, 1)
        x = torch.flatten(x, 1)  # (batch_size, 1280)

        # print(f"\nShape after flattening (corrected): {x.shape}")  # ✅ Debug print
        # print("--")
        return {
            'fmaps': fmaps,
            'features': x,
        }

