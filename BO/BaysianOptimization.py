from global_init import *
from BO.init import *

class BaysianOptimization():
    """
    Class for Baysian Optimization for inputs that trigger FP-exceptions Model
    Attributes
    ----------
    eval_func: GPU function
        The function that are being tested on
    iteration: int
        How many time will the BO be run for
    batch_size: int
        How many time we run the GP before we update it
    acquisition_function: string
        A string indicating which acquisition function to use
        As of right now, we only accept UCB, IP, EI.
    likelihood : The likelihood function
        A function that approximate the likelihood of the GP given new observed datapoint
    bounds: a Tensor with shape 2xd
        The bound of each of the input parameter
    initial_sample: int
        Number of initial data points to sample.
    device:
        The device that the BO will be run on
    Methods
    -------
    initialize_data
    """

    def __init__(self, eval_func: TestFunction, iteration=50, batch_size=5, acquisition_function='ei',
                 likelihood_func=GaussianLikelihood(), bounds: Input_bound=None, device=torch.device("cuda")):
        self.eval_func = eval_func
        self.iteration = iteration
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function
        self.bounds_object = bounds
        self.device = device
        self.likelihood = likelihood_func.to(device=self.device)
        self.time = time.time()
        self.best_bound = None
        self.exception_per_bounds = torch.zeros(self.bounds_object.num_bounds)

        self.results = {}
        self.exception_induced_params = {}
        self.error_types = ["max_inf", "min_inf", "max_under", "min_under", "nan"]
        for type in self.error_types:
            self.results[type] = 0
            self.exception_induced_params[type] = []

        # initialize training data and model
        self.model_params_bounds = {}
        self.initialize_data()
        self.initialize_model()
        self.acq = AcquisitionFunction(self.acquisition_function, kappa=3.0, xi=2.5, fn_type=self.eval_func.fn_type)

    def initialize_data(self, normalize=False):
        """
        :param
        normalize: bool
            If the initial data need to be normalize
        :return: Tuple
            A tuple containing the training data
        """
        initial_x = self.bounds_object.bounds_sampler(10)
        if normalize:
            x_min, x_max = initial_x.min(), initial_x.max()
            new_min, new_max = -1e+100, 1e+100
            initial_x = (initial_x - x_min)/(x_max - x_min)*(new_max - new_min) + new_min
        
        initial_x, initial_y = self.evaluate_candidates(initial_x)
        self.train_y, best_indices = initial_y.unsqueeze(-1).max(dim=1, keepdim=True)
        self.train_y = self.train_y.squeeze(-1)
        self.train_x = torch.gather(initial_x, 1, best_indices.repeat(1,1,initial_x.shape[-1]))
        if len(self.train_x.shape) == 2:
            self.train_x = self.train_x.unsqueeze(-1)
        assert self.train_y.isnan().sum()==0, "training data result in nan"
        self.best_x, self.best_y = self.train_x, self.train_y

    def initialize_model(self, state_dict=None):
        self.likelihood = GaussianLikelihood().to(device=self.device, dtype=dtype)
        self.GP = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device=self.device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.GP)
        # for param_name, param in self.mll.model.named_parameters():
        #     self.model_params_bounds[param_name] = (-10.0, 10.0)
        if state_dict is not None:
            self.GP.load_state_dict(state_dict)
        fit_mll(self.mll, options={"disp": False, "lr": 0.005}, approx_mll=True)

    def evaluate_candidates(self, candidates):
        targets = torch.empty((self.bounds_object.num_bounds,1), dtype=dtype, device=self.device)
        for i,x in enumerate(candidates):
            new_candidate_target = self.eval_func.eval(x[0])
            new_candidate_target = torch.as_tensor(new_candidate_target, device=self.device, dtype=dtype)
            new_candidate_target, exception_found = self.check_exception(x, new_candidate_target)
            if exception_found:
                self.exception_per_bounds[i] += 1
                print("Input belong to bound: ", self.bounds_object.bounds[i])
            candidates[i] = x
            targets[i] = new_candidate_target
        return candidates, targets

    def suggest_new_candidate(self, n_warmup=5000, n_samples=10):
        """
            A function to find the maximum of the acquisition function
            It uses a combination of random sampling (cheap) and the 'AdamW'
            optimization method. First by sampling `n_warmup` (500) points per bound at random,
            and then running AdamW from `n_samples` (5) random starting points per bound.
            Parameters
            ----------
            :param n_warmup:
                number of times to randomly sample the acquisition function
            :param n_samples:
                number of samples to try
            Returns
            -------
            :return: x_max, The arg max of the acquisition function.
            """

        # Warm up with random points
        #x_tries has shape: B*n_warmup x D
        self.likelihood.eval()
        self.GP.eval()
        x_tries = []
        n_warmup = max(n_warmup, round(100000/self.bounds_object.num_bounds))
        warmup_x = self.bounds_object.bounds_sampler(n_warmup)
        with torch.no_grad():
            posterior = self.GP.likelihood(self.GP(warmup_x))
            mean = posterior.mean
            while mean.ndim > 2:
                # take average over batch dims
                mean = mean.mean(dim=0)
            f_pred = mean
            n_best = max(1,  round(warmup_x.shape[1] * 0.05))
            best_idcs = torch.topk(f_pred, n_best).indices.unsqueeze(-1)
            best_X = torch.gather(warmup_x, 1, best_idcs.repeat(1,1,warmup_x.shape[-1]))
        for X, bound in zip(best_X, self.bounds_object.bounds):
            x_tries.append(sample_points_around_best(X, n_discrete_points=n_samples, sigma=1e-3, bounds=bound))
        x_tries = torch.stack(x_tries, dim=0)

        # Use an optimizer to explore the input space more thoroughly
        lb, ub = self.bounds_object.bounds[:,0,:].unsqueeze(1), self.bounds_object.bounds[:,1,:].unsqueeze(1)
        _clamp = partial(columnwise_clamp, lower=lb, upper=ub)
        clamped_candidates = _clamp(x_tries).requires_grad_(True)

        to_minimize = lambda x: -self.acq.forward(self.GP, self.GP.likelihood, x, y_max=self.best_y)
        optimizer = torch.optim.AdamW([clamped_candidates], lr=1e-5)
        i = 0
        stop = False
        stopping_criterion = ExpMAStoppingCriterion(maxiter=50)
        while not stop:
            self.time = time.time()
            i += 1
            with torch.no_grad():
                X = _clamp(clamped_candidates).requires_grad_(True)

            with gpytorch.settings.fast_pred_var():
                loss = to_minimize(X).sum()
            grad_params = torch.autograd.grad(loss, X)[0]
            def assign_grad():
                optimizer.zero_grad()
                clamped_candidates.grad = grad_params
                return loss

            optimizer.step(assign_grad)
            if clamped_candidates.isnan().sum() > 0:
                clamped_candidates = X
                break
            stop = stopping_criterion.evaluate(fvals=loss.detach())
        clamped_candidates, _ = self.acq.extract_best_candidate(clamped_candidates, self.GP, self.GP.likelihood, y_max=self.best_y)
        return clamped_candidates.detach()

    def check_exception(self, param, val):
        # TODO: check exceptions in batch.
        """
            A function to check if the value returned by the GPU function is an FP exception. It also updates the result
            table if an exception is caught
            ----------
            :param param:
                The input parameter to the evaluation function
            :param val:
                The return value of the evaluation function
            Returns
            -------
            :return: a boolean indicating if the value is an FP exception or not
            """
        # Infinity
        if torch.isinf(val):
            logger.info( "parameter {} caused floating point error {}".format(param, val))
            print("input triggered exception: ", param)
            print("exception value: ", val)
            if val < 0.0:
                self.results["min_inf"] += 1
                self.exception_induced_params["min_inf"].append(param)
                val = -1e+307
                # self.save_trials_to_trigger(exp_name)
            else:
                self.results["max_inf"] += 1
                self.exception_induced_params["max_inf"].append(param)
                # self.save_trials_to_trigger(exp_name)
                val = 1e+307
            return val, True

        # Subnormals
        if torch.isfinite(val):
            if val > -2.22e-308 and val < 2.22e-308:
                logger.info( "parameter {} caused subnormal floating point".format(param))
                if val != 0.0 and val != -0.0:
                    if val < 0.0:
                        print("input triggered exception: ", param)
                        print("exception value: ", val)
                        self.results["min_under"] += 1
                        self.exception_induced_params["min_under"].append(param)
                    else:
                        self.results["max_under"] += 1
                        self.exception_induced_params["max_under"].append(param)
                    return val, True

        # Nan
        if torch.isnan(val):
            logger.info( "parameter {} caused floating point error {}".format(param, val))
            print("input triggered exception: ", param)
            print("exception value: ", val)
            self.results["nan"] += 1
            self.exception_induced_params["nan"].append(param)
            val = 2.1219957915e-314
            return val, True
        return val, False

    def time_it(self, execution_name):
        check_point = time.time()-self.time
        self.time = time.time()
        print("{} executed in {}".format(execution_name, check_point))

    def best_interval(self):
        max_index = 0
        if torch.count_nonzero(self.exception_per_bounds) >0:
            max_index = torch.argmax(self.exception_per_bounds)
        else:
            max_y = torch.argmax(self.train_y)
            max_index = torch.div(max_y, self.train_y.shape[-1], rounding_mode="floor")

        self.best_bound = self.bounds_object.bounds[max_index]


    def train(self):
        print("Begin BO")
        start_fitting = time.time()
        for i in range(self.iteration):
            if i % self.batch_size == 0 and i != 0:
                old_state_dict = self.mll.model.state_dict()
                self.initialize_model(state_dict=old_state_dict)
            new_candidates = self.suggest_new_candidate()
            new_candidates, new_targets = self.evaluate_candidates(new_candidates)
            best_new_target, _ = new_targets.max(dim=1, keepdim=True)
            target_mask = torch.gt(best_new_target, self.best_y)
            self.best_y = torch.where(target_mask, best_new_target, self.best_y)
            self.best_x = torch.where(target_mask, new_candidates, self.best_x)
            self.train_x = torch.cat([self.train_x, new_candidates], dim=1)
            self.train_y = torch.cat([self.train_y, new_targets], dim=1)
            assert self.train_x.shape[0] == self.train_y.shape[0], f"shape mismatch, got {self.train_x.shape[0]} for training data but {self.train_y.shape[0]} for testing data"
            self.acq.update_params
        
        self.best_interval()
        print("Fitting time: ", time.time() - start_fitting)