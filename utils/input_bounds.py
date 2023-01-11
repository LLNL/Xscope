import gpytorch
from utils.init import *
from botorch.optim.initializers import sample_perturbed_subset_dims, sample_truncated_normal_perturbations

class Input_bound():
    def __init__(self, split, num_input=1, input_type="fp", device = torch.device("cuda"), input_range = None, f_type=None) -> None:
        self.device = device
        self.input_range = input_range
        if self.input_range is None:
            self.input_range = [-max_normal, max_normal] if input_type=="fp" else [-307,307]
        self.ignore_params = None
        self.bounds = self.generate_bounds(split, num_input)
        self.num_bounds, _, self.dim = self.bounds.shape

    def generate_bounds(self, split, num_input):
        upper_lim = self.input_range[1]
        lower_lim = self.input_range[0]
        assert upper_lim >= lower_lim, f"upper bound{upper_lim} must be greater than lower bound {lower_lim}"
        if split == 1:
            if num_input > 3:
                lower_bound = torch.as_tensor(lower_lim, dtype=torch.float64).repeat(3)
                upper_bound = torch.as_tensor(upper_lim, dtype=torch.float64).repeat(3)
                b = torch.stack((lower_bound,upper_bound)).unsqueeze(dim=0).to(dtype=dtype, device=device)
                params_index = torch.arange(num_input)
                params_combinations = torch.stack(torch.split(params_index, 3))
                self.ignore_params=torch.empty(len(params_combinations), num_input-3)
                for i, split in enumerate(params_combinations):
                    ignore = [index for index in params_index if index not in split]
                    ignore = torch.stack(ignore)
                    self.ignore_params[i] = ignore
            
            else:
                lower_bound = torch.as_tensor(lower_lim, dtype=torch.float64).repeat(num_input)
                upper_bound = torch.as_tensor(upper_lim, dtype=torch.float64).repeat(num_input)
                b = torch.stack((lower_bound,upper_bound)).unsqueeze(dim=0).to(dtype=dtype, device=device)
        else:
            limits = torch.linspace(lower_lim,upper_lim,split+1,dtype=torch.float64)
            ranges = torch.combinations(limits, r=2)
            if num_input == 1:
                b = ranges.unsqueeze(-1)
            elif num_input == 2:
                combinations = torch.arange(ranges.shape[0])
                combinations_indices = torch.combinations(combinations,2, with_replacement=True)
                b= torch.stack([torch.index_select(ranges,dim=0, index=index) for index in combinations_indices]).transpose(1,2)
            
            #Testing on 3 params at a time.
            else:
                combinations = torch.arange(ranges.shape[0])
                combinations_indices = torch.combinations(combinations,3, with_replacement=True)
                b = torch.stack([torch.index_select(ranges,dim=0, index=index) for index in combinations_indices]).transpose(1,2)
                #Splitting into chunk of 3 if there are more than 3 inputs
                if num_input > 3:
                    params_index = torch.arange(num_input)
                    params_combinations = torch.stack(torch.split(params_index, 3))
                    self.ignore_params=torch.empty(len(params_combinations), num_input-3)
                    for i, split in enumerate(params_combinations):
                        ignore = [index for index in params_index if index not in split]
                        ignore = torch.stack(ignore)
                        self.ignore_params[i] = ignore
        return b.to(dtype=dtype, device=self.device)
    
    def bounds_sampler(self, num_sample, padding=False):
        lb = self.bounds[:,0,:].unsqueeze(1)
        ub = self.bounds[:,1,:].unsqueeze(1)
        num_bounds = lb.shape[0]
        sampler = torch.distributions.uniform.Uniform(lb, ub)
        samples = sampler.rsample((num_sample,)).to(dtype=dtype, device=self.device).view(num_bounds, num_sample, self.dim)
        if padding:
            samples = self.add_padding(samples)
        return samples

    def sample_points_around_best(
        self,
        best_X: Tensor,
        n_discrete_points: int,
        sigma: float,
        bounds: Tensor,
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
        if best_X is None:
            return
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
            perm = torch.randperm(perturbed_X.shape[0], device=best_X.device)
            perturbed_X = perturbed_X[perm]
        return perturbed_X

    def update_bound(self, new_bound):
        self.bounds = new_bound
        self.num_bounds, _, self.dim = self.bounds.shape
        self.ignore_params = []

