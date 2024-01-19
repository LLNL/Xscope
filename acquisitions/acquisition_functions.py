from acquisitions.init import *
from botorch.utils.torch import BufferDict
from itertools import combinations
from multipledispatch import dispatch
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

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

        if kind not in ['ucb', 'ei', 'poi','qehvi']:
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

    def forward(self, gp, likelihood, x, y_max= None, ref_point=None):
        with gpytorch.settings.fast_pred_var():
            output = likelihood(gp(x))
        mean, std = output.mean, torch.sqrt(output.variance)
        if self.kind == 'ucb':
            return self._ucb(mean, std, self.kappa, self.fn_type)
        if self.kind == 'ei':
            return self._ei(mean, std, y_max, self.xi, self.fn_type)
        if self.kind == 'poi':
            return self._poi(mean, std, y_max, self.xi)
        else:
            return self._qehvi(mean, ref_point)
        
    def get_best_y(self, mean, std, y_max, x):
        if self.kind == 'ucb':
            predicted_y =  self._ucb(mean, std, self.kappa, self.fn_type)
        elif self.kind == 'ei':
            predicted_y = self._ei(mean, std, y_max, self.xi, self.fn_type)
        elif self.kind == 'poi':
            predicted_y = self._poi(mean, std, y_max, self.xi)
        else:
            #predicted_y = self._qehvi()
            err = "not implemented"
            raise NotImplementedError(err)
        ys = predicted_y.unsqueeze(-1)
        ys = torch.nan_to_num(ys, nan=2.1219957915e-314)
        max_acq, indices = ys.max(dim=1, keepdim=True)
        x_max = torch.gather(x, 1, indices.repeat(1,1,x.shape[-1]))
        return max_acq, x_max

    def extract_best_candidate(self, candidates, gp, likelihood, y_max=None, ref_point=None, batch_size=1):
        with torch.no_grad(), autocast(), gpytorch.settings.fast_pred_var():
            ys = self.forward(gp, likelihood, candidates, y_max=y_max, ref_point=ref_point).unsqueeze(-1)
            ys = torch.nan_to_num(ys, nan=2.1219957915e-314)
            if ref_point is not None:
                ys = ys.transpose(1,0)
            # max_acq, indices = ys.max(dim=1, keepdim=True)
            max_acq, indices = torch.topk(ys.squeeze(dim=-1), batch_size)
            x_max = torch.gather(candidates, 1, indices.unsqueeze(-1).repeat(1,1,candidates.shape[-1]))
        return x_max, max_acq

    @staticmethod
    def _ucb(mean, std, kappa, fn_type):
        if fn_type == "min_under" or fn_type == "max_under":
            mask = torch.eq(mean, -0.0)
            mean = torch.where(mask, -1e-100, mean)
        return mean + kappa * std

    @staticmethod
    def _ei(mean, std, y_max, xi, fn_type):
        if fn_type == "min_under" or fn_type == "max_under":
            mask = torch.eq(mean, 0.0)
            mean = torch.where(mask, -1e-303, mean)
        a = (mean - y_max - xi)
        z = a / std
        if not torch.isfinite(z).all():
            return a + std
        norm = torch.distributions.normal.Normal(0.0, 1.0)
        pdf = 1/torch.sqrt(torch.tensor(2.0*math.pi, device=device, dtype=dtype)) * torch.exp(-z**2/2)
        return a * norm.cdf(z) + std * pdf

    @staticmethod
    def _poi(mean, std, y_max, xi):
        z = (mean - y_max - xi)/std
        norm =  torch.distributions.normal.Normal(0.0, 1.0)
        return norm.cdf(z)

    #This function was derived from BOTorch at: https://github.dev/pytorch/botorch/blob/main/botorch/acquisition/multi_objective/monte_carlo.py
    @staticmethod
    def _qehvi(mean, ref_point):
        """Compute the expected (feasible) hypervolume improvement given MC samples.

        Args:
            means: A `num_bounds x n_samples x d`-dim tensor of inputs.
            ref_point: the reference point for pareto computation.

        Returns:
            A `num_bounds x (model_batch_shape)`-dim tensor of expected hypervolume
            improvement for each batch.
        """
        partitioning = DominatedPartitioning(ref_point=ref_point, Y=mean.transpose(1,0))
        batch_shape, n_samples, num_task = mean.shape
        mean = mean.view(n_samples, batch_shape, 1, num_task)
        cell_bounds = partitioning.get_hypercell_bounds()
        cell_lower_bounds = cell_bounds[0]
        cell_upper_bounds = cell_bounds[1]
        obj = mean
        batch_shape = obj.shape[:-2]
        q = obj.shape[-2]
        indices = list(range(q))
        tkwargs = {"dtype": torch.long, "device": ref_point.device}
        q_subset_indices = BufferDict(
            {
                f"q_choose_{i}": torch.tensor(
                    list(combinations(indices, i)), **tkwargs
                )
                for i in range(1, q + 1)
            }
        )
        areas_per_segment = torch.zeros(
            *batch_shape,
            cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=obj.device,
        )
        cell_batch_ndim = cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            cell_upper_bounds.shape[-2],
            1,
            cell_upper_bounds.shape[-1],
        )
        for i in range(1, q + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            index = q_choose_i.view(-1)
            obj_subsets = obj.index_select(dim=-2, index=index)
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), cell_upper_bounds.view(view_shape)
            )
            # substract cell lower bounds, clamp min at zero
            lengths_i = (
                overlap_vertices - cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i
        # sum over segments and average over MC samples
        return areas_per_segment.sum(dim=-1)

