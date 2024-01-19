from utils.init import *
import traceback
from botorch.optim.initializers import sample_perturbed_subset_dims, sample_truncated_normal_perturbations
# from botorch.optim.utils import (
#     _filter_kwargs,
#     _get_extra_mll_args,
#     create_name_filter,
# )

class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


max_normal = 1e+307
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

# from jax.config import config
# config.update("jax_enable_x64", True)


def sample_points_around_best(
    best_X: Tensor,
    n_discrete_points: int,
    sigma: float,
    bounds: Tensor,
    prob_perturb: Optional[float] = None,
    ) -> Optional[Tensor]:
    r"""Find best points and sample nearby points.

    Args:
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
        qmc=False
    )
    if use_perturbed_sampling:
        perturbed_subset_dims_X = sample_perturbed_subset_dims(
            X=best_X,
            bounds=bounds,
            # ensure that we return n_discrete_points
            n_discrete_points=n_discrete_points - n_trunc_normal_points,
            sigma=sigma,
            prob_perturb=prob_perturb,
            qmc=False
        )
        perturbed_X = torch.cat([perturbed_X, perturbed_subset_dims_X], dim=0)
        # shuffle points
        perm = torch.randperm(perturbed_X.shape[0], device=best_X.device)
        perturbed_X = perturbed_X[perm]
    del best_X
    del bounds
    return perturbed_X

class ResultLogger:
    def __init__(self):
        self.results = {}
        self.random_results = {}
        self.exception_induced_params = {}
        self.bounds_list = []
        self.total_error_per_bound = []
        self.start = 0
        self.execution_time = 0

        self.funcs = ["max_inf", "min_inf", "max_under", "min_under", "nan"]

        for type in self.funcs:
            self.results[type] = 0
            self.exception_induced_params[type] = []

    def start_time(self):
        self.start = time.time()

    def log_time(self):
        self.execution_time = time.time() - self.start

    def log_result(self, bo_errors):
        error_count = 0
        for type in self.funcs:
            self.results[type] += bo_errors[type]
            # result_logger.exception_induced_params[type] += bo.exception_induced_params[type]
            error_count += self.results[type]
        self.total_error_per_bound.append(error_count)

    # def summarize_result(self, func_name):
    #     total_exception = 0
    #     bgrt_bo_data = {'Function': [func_name]}
    #     for type in self.funcs:
    #         print('\t' + type + ": ", self.results[type])
    #         bgrt_bo_data[type] = [self.results[type]]
    #         total_exception += self.results[type]

    #     print('\tTotal Exception: ', total_exception)
    #     bgrt_bo_data.update({'Total Exception': [total_exception],
    #                          'Execution Time': [self.execution_time]})
    #     bgrt_interval_data = {}
    #     bgrt_interval_data['Function'] = [func_name]
    #     for bound, total_error in zip(self.bounds_list, self.total_error_per_bound):
    #         bgrt_interval_data[bound] = [total_error]

    #     bgrt_bo_df = pd.DataFrame(bgrt_bo_data)
    #     bgrt_interval_df = pd.DataFrame(bgrt_interval_data)

    #     return bgrt_bo_df, bgrt_interval_df

    def write_result_to_file(self, file_name, data):
        if isfile(file_name):
            data.to_csv(file_name, mode='a', index=False, header=False)
        else:
            data.to_csv(file_name, index=False)
