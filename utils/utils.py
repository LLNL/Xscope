from utils.init import *
from botorch.optim.initializers import sample_perturbed_subset_dims, sample_truncated_normal_perturbations
from botorch.optim.utils import (
    _filter_kwargs,
    _get_extra_mll_args,
    create_name_filter,
)

class OptimizationIteration(NamedTuple):
    itr: int
    fun: float
    time: float


max_normal = 1e+307
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

from jax.config import config
config.update("jax_enable_x64", True)


def sample_points_around_best(
    best_X: Tensor,
    n_discrete_points: int,
    sigma: float,
    bounds: Tensor,
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

def fit_mll(
    mll: MarginalLogLikelihood,
    bounds: Optional[ParameterBounds] = None,
    optimizer_cls: Optimizer = AdamW,
    options: Optional[Dict[str, Any]] = None,
    track_iterations: bool = False,
    approx_mll: bool = False,
) -> Tuple[MarginalLogLikelihood, Dict[str, Union[float, List[OptimizationIteration]]]]:
    r"""Fit a gpytorch model by maximizing MLL with a torch optimizer.

    The model and likelihood in mll must already be in train mode.
    Note: this method requires that the model has `train_inputs` and `train_targets`.

    Args:
        mll: MarginalLogLikelihood to be maximized.
        bounds: A ParameterBounds dictionary mapping parameter names to tuples
            of lower and upper bounds. Bounds specified here take precedence
            over bounds on the same parameters specified in the constraints
            registered with the module.
        optimizer_cls: Torch optimizer to use. Must not require a closure.
        options: options for model fitting. Relevant options will be passed to
            the `optimizer_cls`. Additionally, options can include: "disp"
            to specify whether to display model fitting diagnostics and "maxiter"
            to specify the maximum number of iterations.
        track_iterations: Track the function values and wall time for each
            iteration.
        approx_mll: If True, use gpytorch's approximate MLL computation (
            according to the gpytorch defaults based on the training at size).
            Unlike for the deterministic algorithms used in fit_gpytorch_scipy,
            this is not an issue for stochastic optimizers.

    Returns:
        2-element tuple containing
        - mll with parameters optimized in-place.
        - Dictionary with the following key/values:
        "fopt": Best mll value.
        "wall_time": Wall time of fitting.
        "iterations": List of OptimizationIteration objects with information on each
        iteration. If track_iterations is False, will be empty.

    Example:
        >>> gp = SingleTaskGP(train_X, train_Y)
        >>> mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        >>> mll.train()
        >>> fit_gpytorch_torch(mll)
        >>> mll.eval()
    """
    optim_options = {"maxiter": 100, "disp": True, "lr": 1e-5}
    optim_options.update(options or {})
    exclude = optim_options.pop("exclude", None)
    if exclude is None:
        mll_params = list(mll.parameters())
    else:
        mll_params = [
            v for k, v in filter(create_name_filter(exclude), mll.named_parameters())
        ]

    optimizer = optimizer_cls(
        params=[{"params": mll_params}],
        **_filter_kwargs(optimizer_cls, **optim_options),
    )

    # get bounds specified in model (if any)
    bounds_: ParameterBounds = {}
    if hasattr(mll, "named_parameters_and_constraints"):
        for param_name, _, constraint in mll.named_parameters_and_constraints():
            if constraint is not None and not constraint.enforced:
                bounds_[param_name] = constraint.lower_bound, constraint.upper_bound

    # update with user-supplied bounds (overwrites if already exists)
    if bounds is not None:
        bounds_.update(bounds)

    iterations = []
    t1 = monotonic()

    param_trajectory: Dict[str, List[Tensor]] = {
        name: [] for name, param in mll.named_parameters()
    }
    loss_trajectory: List[float] = []
    i = 0
    stop = False
    stopping_criterion = ExpMAStoppingCriterion(
        **_filter_kwargs(ExpMAStoppingCriterion, **optim_options)
    )
    train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
    while not stop:
        optimizer.zero_grad()
        with gpt_settings.fast_computations(log_prob=approx_mll):
            output = mll.model(*train_inputs)
            # we sum here to support batch mode
            args = [output, train_targets] + _get_extra_mll_args(mll)
            loss = -mll(*args).sum()
            loss.backward()
        loss_trajectory.append(loss.item())
        # for name, param in mll.named_parameters():
        #     param.grad.data.clamp_(-1e+50, 1e+50)
        #     param_trajectory[name].append(param.detach().clone())

        if track_iterations:
            iterations.append(OptimizationIteration(i, loss.item(), monotonic() - t1))

        optimizer.step()
        # project onto bounds:
        if bounds_:
            for pname, param in mll.named_parameters():
                if pname in bounds_:
                    param.data = param.data.clamp(*bounds_[pname])
        i += 1
        stop = stopping_criterion.evaluate(fvals=loss.detach())
    info_dict = {
        "fopt": loss_trajectory[-1],
        "wall_time": monotonic() - t1,
        "iterations": iterations,
    }
    return mll, info_dict

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

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    print(memory_use_info)