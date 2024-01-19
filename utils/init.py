import torch
from os.path import isfile
from time import monotonic
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch import settings as gpt_settings
import time
from models.exactGP import ExactGPModel

from botorch.optim import ExpMAStoppingCriterion
# from botorch.optim.utils import  _get_extra_mll_args

from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torch import Tensor
from torch.cuda.amp import autocast

import numpy as np
import subprocess as sp

from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_normal = 1e+307