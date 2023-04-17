import gpytorch
from models.baseGP import BaseGPModel
import torch

class ExactGPModel(BaseGPModel):
    """
    Class for Gaussian Process Model
    Attributes
    ----------
    train_x: a Torch tensor with shape b x n x d
        The training data
    train_x: a Torch tensor with shape b x n x 1
        The label
    likelihood : The likelihood function
        A function that approximate the likelihood of the GP given new observed datapoint
    Methods
    -------
    forward
    """

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        batch_dim = train_x.shape[0]
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.MaternKernel(nu=2.5,batch_shape=torch.Size([batch_dim])))

        # self.covar_module = gpytorch.kernels.MaternKernel(nu=0.5, batch_shape=torch.Size([batch_dim]))
        #self.covar_module = gpytorch.kernels.MaternKernel(nu=2.5)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)