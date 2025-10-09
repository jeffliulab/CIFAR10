import torch.nn as nn
import torchvision.models as models

class VGG13(nn.Module):
    """
    VGG-13 from Very Deep Convolutional Networks for Large-Scale Image Recognition
    """
    def __init__(self, num_classes=10, input_channels=3, pretrained=False):
        super().__init__()

        self.model = models.vgg13(pretrained=pretrained)

        if input_channels != 3:
            self.model.features[0] = nn.Conv2d(
                input_channels, 64, kernel_size=3, stride=1, padding=1
            )
            
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        in_features = 512
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        return self.model(x)