
#!/usr/bin/env python3
import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights

model_name = "resnet50"
model = models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
dummy_input = torch.randn(1, 3, 224, 224)
torch_out = model(dummy_input)
torch.onnx.export(model, dummy_input, model_name+".onnx",
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}},
                  verbose=False)
