import torch.nn as nn

class TwoLayerMLP(nn.Module):
    """
    Two-layer MLP as described in Experiment 2 of "Adam: A Method for Stochastic Optimization"
    Fill in all TODOs to complete the network.
    """
    def __init__(self,
                 input_size: int = 784,
                 hidden_size: int = 1024,
                 num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            # Linear layer (input_size -> hidden_size)
            nn.Linear(input_size, hidden_size),
            # Activation
            nn.ReLU(),
            # Linear layer (hidden_size -> hidden_size)
            nn.Linear(hidden_size, hidden_size),
            # Activation
            nn.ReLU(),
            # Output linear layer (hidden_size -> num_classes)
            nn.Linear(hidden_size, num_classes),
        )


    def forward(self, x):
        out = self.net(x)
        return out


# ====================================================================
# 附加代码：直接运行本程序，可以将model导出为 ONNX format
# ====================================================================
import torch.onnx

if __name__ == '__main__':
    # 1. Instantiate the model from your class
    model = TwoLayerMLP()
    
    # 2. Create a dummy input tensor with the expected shape 
    # (batch_size, channels, height, width)
    # ❗️ THIS IS THE CORRECTED LINE ❗️
    # This model expects a 28x28 grayscale image, which flattens to 784 features.
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # 3. Define the path for the output ONNX file
    onnx_model_path = "TwoLayerMLP.onnx"

    # 4. Export the model to the specified file
    torch.onnx.export(model, 
                      dummy_input, 
                      onnx_model_path, 
                      verbose=False)

    print(f"✅ Model successfully exported to: {onnx_model_path}")
    print("Next step: Go to https://netron.app and upload this file to visualize it.")
