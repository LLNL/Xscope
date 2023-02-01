import gpytorch
from abc import ABC, abstractmethod

class BaseGPModel(ABC, gpytorch.models.ExactGP):
    """
    Class for the base Gaussian Process Model
    Attributes
    ----------
    train_x: a Torch tensor with shape n x d
        The training data
    train_x: a Torch tensor with shape n x 1
        The label
    likelihood : The likelihood function
        A function that approximate the likelihood of the GP given new observed datapoint
    Methods
    -------
    forward
    """

    def __init__(self, train_x, train_y, likelihood):
        super(BaseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = None
        self.covar_module = None

    @abstractmethod
    def forward(self, x):
        pass
