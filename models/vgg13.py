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

# ====================================================================
# 附加代码：直接运行本程序，可以将model导出为 ONNX format
# ====================================================================
import torch.onnx

if __name__ == '__main__':
    # 1. Instantiate the model from your class
    # Using default parameters: num_classes=10, input_channels=3
    model = VGG13()
    
    # 2. Create a dummy input tensor with a standard shape for VGG
    # (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 3. Define the path for the output ONNX file
    onnx_model_path = "vgg13.onnx"

    # 4. Export the model to the specified file
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_model_path, 
                      verbose=False)

    print(f"✅ Model successfully exported to: {onnx_model_path}")
    print("Next step: Go to https://netron.app and upload this file to visualize it.")
