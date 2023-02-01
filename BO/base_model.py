from global_init import *
from BO.init import *
from abc import ABC, abstractmethod

class bo_base(ABC):
    def __init__(self, eval_func: TestFunction, iteration=50, batch_size=5, acquisition_function='ei', bounds: Input_bound=None, device=torch.device("cuda")):
        self.eval_func = eval_func
        self.iteration = iteration
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function
        self.bounds_object = bounds
        self.device = device
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
        self.acq = AcquisitionFunction(self.acquisition_function, kappa=3.0, xi=2.5, fn_type=self.eval_func.fn_type)
        self.likelihood = None
        self.GP = None
        self.mll = None

    @abstractmethod
    def initialize_data(self, normalize, num_samples):
        """
        :param
        normalize: bool
            If the initial data need to be normalize
        :return: Tuple
            A tuple containing the training data
        """
        pass

    @abstractmethod
    def initialize_model(self, state_dict):
        """
        :param
        state_dict: dictionary
            The weights of the surrogate model
        """
        pass

    @abstractmethod
    def suggest_new_candidate(self, n_warmup, n_samples):
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
        pass

    @abstractmethod
    def evaluate_candidates(self, candidates):
        """
        A function to calculate the targets value given the candidates and check for exceptions.
        If there are exceptions, it will report the exceptions and assign a new 
        values to the targets.
        Returns
        -------
        :return: candidates, The inputs to the blackbox function.
                 targets, The output from the blackbox function given the candidates.
        """
        pass

    @abstractmethod
    def train(self):
        pass

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

    def thorough_space_exploration(self, candidates):
        """
        A function that uses an "AdamW" optimizer to search for better inputs.
        -------
        :param candidates:
            The candidates to be optimized.
        Returns
        -------
        :return: A new set of predicted best candidates for each bounds.

        """
        lb, ub = self.bounds_object.bounds[:,0,:].unsqueeze(1), self.bounds_object.bounds[:,1,:].unsqueeze(1)
        _clamp = partial(columnwise_clamp, lower=lb, upper=ub)
        clamped_candidates = _clamp(candidates).requires_grad_(True)
        to_minimize = lambda x: -self.acq.forward(self.GP, self.GP.likelihood, x, y_max=self.best_y)
        optimizer = torch.optim.AdamW([clamped_candidates], lr=1e-5)
        i = 0
        stop = False
        stopping_criterion = ExpMAStoppingCriterion(maxiter=50)
        while not stop:
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
        return clamped_candidates

    def best_interval(self):
        max_index = 0
        if torch.count_nonzero(self.exception_per_bounds) >0:
            max_index = torch.argmax(self.exception_per_bounds)
        else:
            max_y = torch.argmax(self.train_y)
            max_index = torch.div(max_y, self.train_y.shape[-1], rounding_mode="floor")

        self.best_bound = self.bounds_object.bounds[max_index]