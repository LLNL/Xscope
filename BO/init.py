from functools import partial
import logging
import gpytorch

import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.optim import ExpMAStoppingCriterion
from botorch.optim.utils import columnwise_clamp
from botorch.optim.fit import ParameterBounds
from botorch.utils.multi_objective.pareto import is_non_dominated

from torch.cuda.amp import autocast
from acquisitions.acquisition_functions import AcquisitionFunction
from utils.input_bounds import Input_bound

from utils.utils import fit_mll, sample_points_around_best
from test_function import TestFunction
from BO.base_model import bo_base

from utils.input_bounds import Input_bound
import time

dtype = torch.float64
logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)