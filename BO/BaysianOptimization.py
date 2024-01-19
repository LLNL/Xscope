from global_init import *
from BO.init import *
from models.exactGP import ExactGPModel
from gpytorch.likelihoods import GaussianLikelihood
from botorch.fit import fit_gpytorch_mll
import warnings

class BaysianOptimization(bo_base):
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
    def __init__(self, eval_func: TestFunction, iteration=10, batch_size=5, acquisition_function='ei',
                 bounds: Input_bound=None, device=torch.device("cuda")):
        super(BaysianOptimization, self).__init__(eval_func, iteration, batch_size, acquisition_function, bounds, device)
        # initialize training data and model
        self.initialize_data()
        self.initialize_model()


    def initialize_data(self, normalize=False):
        """
        :param
        normalize: bool
            If the initial data need to be normalize
        :return: Tuple
            A tuple containing the training data
        """
        initial_x = self.bounds_object.bounds_sampler(5)
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
        del initial_x
        del initial_y
        self.best_x, self.best_y = self.train_x, self.train_y

    def initialize_model(self, state_dict=None):
        # self.train_x = self.train_x.flatten(start_dim=0, end_dim=1)
        # self.train_y = self.train_y.flatten()
        self.likelihood = GaussianLikelihood().to(device=self.device, dtype=dtype)
        self.GP = ExactGPModel(self.train_x, self.train_y, self.likelihood).to(device=self.device, dtype=dtype)
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.GP)
        if state_dict is not None:
            self.GP.load_state_dict(state_dict)
        try:
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(self.mll, optimizer=botorch.optim.fit.fit_gpytorch_torch)
        except:
            exit

    def evaluate_candidates(self, candidates):
        num_candidates = candidates.shape[1]
        targets = torch.empty((self.bounds_object.num_bounds,num_candidates), dtype=dtype, device=self.device)
        for i,candidates_per_bound in enumerate(candidates):
            for j, candidate in enumerate(candidates_per_bound):
                # eval_time_one_candidate = time.time()
                new_candidate_target = self.eval_func.eval(candidate)
                # print("eval one candidate: ", time.time()- eval_time_one_candidate)
                new_candidate_target = torch.as_tensor(new_candidate_target, device=self.device, dtype=dtype)
                new_candidate_target, exception_found = self.check_exception(candidate, new_candidate_target)
                if exception_found:
                    self.exception_per_bounds[i] += 1
                    # print("Input belong to bound: ", self.bounds_object.bounds[i])
                targets[i][j] = new_candidate_target
                del new_candidate_target

        return candidates, targets

    def suggest_new_candidate(self, n_warmup=250, n_samples=10):
        """
            A function to find the maximum of the acquisition function
            It uses a combination of random sampling (cheap) and the 'AdamW'
            optimization method. First by sampling `n_warmup` (300) points per bound at random,
            and then running AdamW from `n_samples` (10) random starting points per bound.
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
        # n_warmup = 300
        warmup_x = self.bounds_object.bounds_sampler(n_warmup)
        # warmup_x = warmup_x.flatten(start_dim=0, end_dim=1)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(True), gpytorch.settings.linalg_dtypes(default=torch.half), gpytorch.settings.skip_posterior_variances(state=True),gpytorch.settings.memory_efficient(state=True):
        # , gpytorch.settings.linalg_dtypes(default=torch.half), gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True), gpytorch.settings.max_cholesky_size(0), gpytorch.settings.skip_posterior_variances(state=True):
            posterior = self.likelihood(self.GP(warmup_x))
            mean = posterior.mean
        while mean.ndim > 2:
            # take average over batch dims
            mean = mean.mean(dim=0)
        f_pred = torch.nan_to_num(mean)
        n_best = max(1,  round(warmup_x.shape[1] * 0.05))
        best_idcs = torch.topk(f_pred, n_best).indices.unsqueeze(-1)
        best_X = torch.gather(warmup_x, 1, best_idcs.repeat(1,1,warmup_x.shape[-1]))
        best_X = torch.nan_to_num(best_X)
        best_X = torch.transpose(best_X, 1,0).reshape(best_X.shape[1], -1)
        bounds_T = torch.transpose(self.bounds_object.current_bounds,1,0).reshape(2, -1)
        x_tries = sample_points_around_best(best_X, n_discrete_points=n_samples, sigma=1e-3, bounds = bounds_T)
        # for X, bound in zip(best_X, self.bounds_object.current_bounds):
        #     x_tries.append(sample_points_around_best(X, n_discrete_points=n_samples, sigma=1e-3, bounds=bound))
        # x_tries = torch.stack(x_tries, dim=0)
        x_tries = x_tries.reshape(n_samples, self.bounds_object.current_bounds.shape[0], -1).transpose(1,0)
        candidates = self.thorough_space_exploration(x_tries)
        del best_X, x_tries, mean, f_pred, warmup_x, posterior, bounds_T
        return candidates.detach()

    def train(self):

        for i in range(self.iteration):
            # torch.cuda.empty_cache()
            new_candidates = self.suggest_new_candidate()
            # print("inference time: ", time.time()-inference_begin)
            # iteration_eval = time.time()
            new_candidates, new_targets = self.evaluate_candidates(new_candidates)
            # print("iteration eval: ", time.time()- iteration_eval)
            self.train_x = torch.cat([self.train_x, new_candidates], dim=1)
            self.train_y = torch.cat([self.train_y, new_targets], dim=1)
            # assert self.train_x.shape[0] == self.train_y.shape[0], f"shape mismatch, got {self.train_x.shape[0]} for training data but {self.train_y.shape[0]} for testing data"
            old_state_dict = self.mll.model.state_dict()
            self.initialize_model(state_dict=old_state_dict)
            del old_state_dict
            self.acq.update_params()
            del new_candidates
            del new_targets
        
        self.best_interval()
        del self.mll, self.likelihood, self.GP, self.acq, self.train_x, self.train_y