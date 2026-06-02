# environment-check.py
import sys
import numpy as np
import torch
import matplotlib

print("Python executable:", sys.executable)
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("Matplotlib:", matplotlib.__version__)