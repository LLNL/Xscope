# from distrax import Normal
import gpytorch
import torch
from torch.cuda.amp import autocast
import math

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
