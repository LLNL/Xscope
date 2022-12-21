import gpytorch
from utils.init import *
from botorch.optim.initializers import sample_perturbed_subset_dims, sample_truncated_normal_perturbations

class Input_bound():
    def __init__(self, split="many", num_input=1, input_type="fp", device = torch.device("cuda"), input_range = None, f_type=None) -> None:
        self.device = device
        self.input_range = input_range
        self.bounds = self.generate_bounds(split, num_input, input_type, f_type)
        self.num_bounds, _, self.dim = self.bounds.shape
        self.padded_x = torch.ones(self.dim, dtype=dtype, device=self.device)
        self.removed_bounds = torch.zeros(self.num_bounds)

    def generate_bounds(self, split, num_input, input_type="fp", f_type=None):
        b = []
        if self.input_range == None:
            upper_lim = max_normal
            lower_lim = -max_normal
        else:
            upper_lim = self.input_range[1]
            lower_lim = self.input_range[0]
            assert upper_lim >= lower_lim, f"upper bound{upper_lim} must be greater than lower bound {lower_lim}"
        if input_type == "exp":
            upper_lim = 307
            lower_lim = -307
        lower_bound = torch.as_tensor(lower_lim, dtype=torch.float64).repeat(num_input)
        upper_bound = torch.as_tensor(upper_lim, dtype=torch.float64).repeat(num_input)
        if split == 1:
            b = torch.stack((lower_bound,upper_bound)).unsqueeze(dim=0).to(dtype=dtype, device=device)
        elif split == 2:
            mid_split = (lower_lim + upper_lim)/2.0
            mid_split = torch.tensor([mid_split], dtype=torch.float64).repeat(num_input)
            lower_half = torch.stack((lower_bound, mid_split))
            upper_half = torch.stack((mid_split, upper_bound))
            b = torch.stack((lower_half, upper_half)).to(dtype=dtype, device=device)
        else:
            limits = torch.linspace(lower_lim,upper_lim,split,dtype=torch.float64)
            # if f_type=="max_under" or f_type=="min_under":
            #     # extra_range = torch.tensor([1e-307, 1e-100, 0.0, -1e-307, -1e-100], dtype=torch.float64)
            #     extra_range = torch.tensor([0.0], dtype=torch.float64)
            #     limits = torch.cat([limits, extra_range])
            #     limits, _ = torch.sort(limits)
            ranges = []
            if input_type == "exp":
                for i in range(len(limits) - 1):
                    x = limits[i]
                    y = limits[i + 1]
                    t = (min(x, y), max(x, y))
                    ranges.append(t)
            else:
                for i in range(len(limits) - 1):
                    x = limits[i]
                    y = limits[i + 1]
                    t = [[min(x, y)], [max(x, y)]]
                    ranges.append(t)
            if num_input == 1:
                for r1 in ranges:
                    b.append(torch.tensor([r1], dtype=dtype, device=self.device).squeeze(0))
            elif num_input == 2:
                for r1 in ranges:
                    for r2 in ranges:
                        bound = torch.transpose(torch.tensor([r1,r2], dtype=dtype, device=self.device).squeeze(),0,1)
                        b.append(bound)
            else:
                for r1 in ranges:
                    for r2 in ranges:
                        bound = torch.transpose(torch.tensor([r1,r2,r2], dtype=dtype, device=self.device).squeeze(),0,1)
                        b.append(bound)
            b = torch.stack(b, dim=0)
            print(b.shape)
        return b
    
    def bounds_sampler(self, num_sample, padding=False):
        active_bounds = self.bounds[self.removed_bounds==0]
        lb = active_bounds[:,0,:].unsqueeze(1)
        ub = active_bounds[:,1,:].unsqueeze(1)
        num_bounds = lb.shape[0]
        sampler = torch.distributions.uniform.Uniform(lb, ub)
        samples = sampler.rsample((num_sample,)).to(dtype=dtype, device=self.device).view(num_bounds, num_sample, self.dim)
        if padding:
            samples = self.add_padding(samples)
        return samples

    def sample_around_best(self, best_candidates, num_samples=10000, padding=False):
        active_bounds = self.bounds[self.removed_bounds==0]
        lb = active_bounds[:,0,:].unsqueeze(1)
        ub = active_bounds[:,1,:].unsqueeze(1)
        lb_dist = torch.abs(best_candidates - torch.abs(lb))
        ub_dist = torch.abs(best_candidates - torch.abs(ub))
        mask_dist = torch.lt(lb_dist, ub_dist)
        std = torch.where(mask_dist, lb_dist, ub_dist)
        samples = []
        for _ in range(num_samples):
            samples.append(torch.normal(best_candidates, std))
        samples = torch.cat(samples, dim=1).to(dtype=dtype, device=self.device)
        if padding:
            samples = self.add_padding(samples)
        return samples.detach()

    def sample_points_around_best(
        self,
        X: Tensor,
        gp: ExactGPModel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        n_discrete_points: int,
        sigma: float,
        bounds: Tensor,
        best_pct: float = 5.0,
        subset_sigma: float = 1e-1,
        prob_perturb: Optional[float] = None,
        ) -> Optional[Tensor]:
        r"""Find best points and sample nearby points.

        Args:
            acq_function: The acquisition function.
            n_discrete_points: The number of points to sample.
            sigma: The standard deviation of the additive gaussian noise for
                perturbing the best points.
            bounds: A `2 x d`-dim tensor containing the bounds.
            best_pct: The percentage of best points to perturb.
            subset_sigma: The standard deviation of the additive gaussian
                noise for perturbing a subset of dimensions of the best points.
            prob_perturb: The probability of perturbing each dimension.

        Returns:
            An optional `n_discrete_points x d`-dim tensor containing the
                sampled points. This is None if no baseline points are found.
        """
        if X is None:
            return
        with torch.no_grad():
            posterior = likelihood(gp(X))
            mean = posterior.mean
            while mean.ndim > 2:
                # take average over batch dims
                mean = mean.mean(dim=0)
            f_pred = mean
            if f_pred.shape[-1] == 1:
                f_pred = f_pred.squeeze(-1)
            n_best = max(1, round(X.shape[0] * best_pct / 100))
            # the view() is to ensure that best_idcs is not a scalar tensor
            best_idcs = torch.topk(f_pred, n_best).indices.view(-1)
            best_X = X[best_idcs]
        use_perturbed_sampling = best_X.shape[-1] >= 20 or prob_perturb is not None
        n_trunc_normal_points = (
            n_discrete_points // 2 if use_perturbed_sampling else n_discrete_points
        )
        perturbed_X = sample_truncated_normal_perturbations(
            X=best_X,
            n_discrete_points=n_trunc_normal_points,
            sigma=sigma,
            bounds=bounds,
        )
        if use_perturbed_sampling:
            perturbed_subset_dims_X = sample_perturbed_subset_dims(
                X=best_X,
                bounds=bounds,
                # ensure that we return n_discrete_points
                n_discrete_points=n_discrete_points - n_trunc_normal_points,
                sigma=sigma,
                prob_perturb=prob_perturb,
            )
            perturbed_X = torch.cat([perturbed_X, perturbed_subset_dims_X], dim=0)
            # shuffle points
            perm = torch.randperm(perturbed_X.shape[0], device=X.device)
            perturbed_X = perturbed_X[perm]
        return perturbed_X



    def add_padding(self, candidates):
        i = 0
        padded_candidates = torch.empty(self.num_bounds, candidates.shape[1], self.dim, dtype=dtype, device=self.device)
        for index in range(len(padded_candidates)):
            if self.removed_bounds[index] == 1:
                padded_candidates[index] = self.padded_x
            else:
                padded_candidates[index] = candidates[i]
                i += 1
        return padded_candidates

    def remove_bound(self, index):
        self.removed_bounds[index] = 1

    def set_padded_y(self, padded_y_value):
        self.padded_y = padded_y_value