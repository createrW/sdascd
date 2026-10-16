"""Print reproduction environment without shell-dependent inline Python quoting."""
import sys

import torch
import torchvision
import transformers

print(sys.version)
print(torch.__version__, torchvision.__version__, transformers.__version__)
print("CUDA:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU/MPS")
