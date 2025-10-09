import torch.nn as nn

class ThreeLayerConv(nn.Module):
    """
    Three-layer CNN as described in Experiment 3 of 
    "Adam: A Method for Stochastic Optimization":

    Our CNN architecture has three alternating stages of 5x5
    convolution filters and 3x3 max pooling with stride of 2 
    that are followed by a fully connected layer of 1000 ReLU units.
    """
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # First conv layer (input_channels -> 64), 5x5 kernel, padding=2
            nn.Conv2d(input_channels, 64, kernel_size=5, padding=2),  # 32x32 -> 32x32
            # ReLU activation
            nn.ReLU(),
            # First max pooling layer (3x3 kernel, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 32x32 -> 15x15
            # Second conv layer (64 -> 64), 5x5 kernel, padding=2
            nn.Conv2d(64, 64, kernel_size=5, padding=2),              # 15x15 -> 15x15
            # ReLU activation
            nn.ReLU(),
            # Second max pooling layer (3x3 kernel, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 15x15 -> 7x7
            # Third conv layer (64 -> 128), 5x5 kernel, padding=2
            nn.Conv2d(64, 128, kernel_size=5, padding=2),             # 7x7 -> 7x7
            # ReLU activation
            nn.ReLU(),
            # Third max pooling layer (3x3 kernel, stride=2)
            nn.MaxPool2d(kernel_size=3, stride=2),                    # 7x7 -> 3x3
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ====================================================================
# 附加代码：直接运行本程序，可以将model导出为 ONNX format
# ====================================================================
import torch.onnx

if __name__ == '__main__':
    # 1. Instantiate the model from your class
    model = ThreeLayerConv()
    
    # 2. Create a dummy input tensor with the expected shape 
    # (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # 3. Define the path for the output ONNX file
    onnx_model_path = "three_layer_conv.onnx"

    # 4. Export the model to the specified file
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_model_path, 
                      verbose=False)

    print(f"✅ Model successfully exported to: {onnx_model_path}")
    print("Next step: Go to https://netron.app and upload this file to visualize it.")
