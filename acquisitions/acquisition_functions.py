from acquisitions.init import *

"""
This code is inspired by this github: https://github.com/fmfn/BayesianOptimization
"""
class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi, fn_type, kappa_decay=1, kappa_decay_delay=0):
        self.kappa = kappa
        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay

        self.xi = xi

        self._iters_counter = 0
        self.fn_type = fn_type

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def forward(self, gp, likelihood, x, y_max= None):
        if self.kind == 'ucb':
            return self._ucb(gp, likelihood, x, self.kappa, self.fn_type)
        if self.kind == 'ei':
            return self._ei(gp, likelihood, x, y_max, self.xi, self.fn_type)
        if self.kind == 'poi':
            return self._poi(gp, likelihood, x, y_max, self.xi)

    def extract_best_candidate(self, candidates, gp, likelihood, y_max=None):
        with torch.no_grad(), autocast(), gpytorch.settings.fast_pred_var():
            ys = self.forward(gp, likelihood, candidates, y_max=y_max).unsqueeze(-1)
            ys = torch.nan_to_num(ys, nan=2.1219957915e-314)
            max_acq, indices = ys.max(dim=1, keepdim=True)
            x_max = torch.gather(candidates, 1, indices.repeat(1,1,candidates.shape[-1]))
        return x_max, max_acq

    @staticmethod
    def _ucb(gp, likelihood, x, kappa, fn_type):
        with gpytorch.settings.fast_pred_var():
            output = likelihood(gp(x))
        mean, std = output.mean, torch.sqrt(output.variance)
        if fn_type == "min_under" or fn_type == "max_under":
            mask = torch.eq(mean, -0.0)
            mean = torch.where(mask, -1e-100, mean)
        return mean + kappa * std

    @staticmethod
    def _ei(gp, likelihood, x, y_max, xi, fn_type):
        with gpytorch.settings.fast_pred_var():
            output = likelihood(gp(x))
        mean, std = output.mean, torch.sqrt(output.variance)
        if fn_type == "min_under" or fn_type == "max_under":
            mask = torch.eq(mean, 0.0)
            mean = torch.where(mask, -1e-303, mean)
        a = (mean - y_max - xi)
        z = a / std
        if not torch.isfinite(z).all():
            return a + std
        norm = torch.distributions.normal.Normal(torch.tensor([0.0]).to(device=device, dtype=dtype), torch.tensor([1.0]).to(device=device, dtype=dtype))
        pdf = 1/torch.sqrt(torch.tensor(2.0*math.pi).to(device=device, dtype=dtype)) * torch.exp(-z**2/2)
        return a * norm.cdf(z) + std * pdf

    @staticmethod
    def _poi(gp, likelihood, x, y_max, xi):
        with gpytorch.settings.fast_pred_var():
            output = likelihood(gp(x))
        mean, std = output.mean, torch.sqrt(output.variance)
        z = (mean - y_max - xi)/std
        norm = Normal(torch.tensor([0.0]).to(device=device, dtype=dtype), torch.tensor([1.0]).to(device=device, dtype=dtype))
        return norm.cdf(z)