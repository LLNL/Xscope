import logging

import jax.numpy as jnp
import jax.random as jr
from jax import jit, vmap, grad
from tqdm.auto import tqdm

import gpjax as gpx

import optax as ox

from sklearn.preprocessing import PowerTransformer, MinMaxScaler
from utils import *

logging.basicConfig(filename='Xscope.log', level=logging.INFO)
logger = logging.getLogger(__name__)
max_normal = 1e+307
key = jr.PRNGKey(250)

class GPModel():
    """
    Class for Gaussian Process Model
    Attributes
    ----------
    dataset: a dataset containing the training data and target.
    Methods
    -------
    fit
    predict
    """
    def __init__(self, dataset):
        self.D = dataset
        self.kernel = gpx.Matern12()
        self.prior = gpx.Prior(kernel=self.kernel)
        self.likelihood = gpx.Gaussian(num_datapoints=self.D.n)
        self.posterior = self.prior * self.likelihood
        self.parameter_state = gpx.initialise(self.posterior, key)
        self.params, self.trainable, self.bijectors = self.parameter_state.unpack()
        # Exponential decay of the learning rate.
        scheduler = ox.exponential_decay(
            init_value=1e-5, 
            transition_steps=1000,
            decay_rate=0.99)

        # Combining gradient transforms using `optax.chain`.
        self.opt = ox.chain(
            ox.clip_by_global_norm(5.0),  # Clip by the gradient by the global norm.
            ox.scale_by_adam(),  # Use the updates from adam.
            ox.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            ox.scale(-1.0)
        )

    def fit(self, dataset=None):
        if dataset is not None:
            self.D(dataset)
        self.mll = jit(self.posterior.marginal_log_likelihood(self.D, negative=True))
        inference_state = gpx.fit(self.mll,self.parameter_state,self.opt,n_iters=50,)
        self.params, _ = inference_state.unpack()

    def predict(self, x):
        latent_dist = self.posterior(self.params, self.D)(x)
        predictive_dist = self.likelihood(self.params ,latent_dist)
        return predictive_dist.mean(), predictive_dist.stddev()

class BO_JAX():
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

    def __init__(self, eval_func, iteration=25, batch_size=5, acquisition_function='ucb', bounds=None, initial_sample=1):
        self.eval_func = eval_func
        self.iteration = iteration
        self.batch_size = batch_size
        self.acquisition_function = acquisition_function
        self.bounds = bounds
        self.lower_bounds = jnp.expand_dims(bounds[:,0,:], axis=1)
        self.upper_bounds =  jnp.expand_dims(bounds[:,1,:], axis=1)
        self.num_bounds, _, self.dim = self.bounds.shape

        self.results = {}
        self.exception_induced_params = {}
        self.error_types = ["max_inf", "min_inf", "max_under", "min_under", "nan"]
        for type in self.error_types:
            self.results[type] = 0
            self.exception_induced_params[type] = []

        # initialize training data and model
        # self.model_params_bounds = {}
        self.initialize_data(initial_sample)
        self.initialize_model()
        self.acq = UtilityFunction(self.acquisition_function, kappa=2.5, xi=0.1e-1)
        self.optim = AdamW
        self.suggest_new_candidate()

    def initialize_data(self, num_sample=5, normalize=True):
        """
        :param
        num_sample: int
            The number of sample to initialize
        :return: Tuple
            A tuple containing the training data
        """
        
        initial_X = self.random_sample(num_sample)
        if normalize:
            pt = MinMaxScaler()
            initial_X = pt.fit_transform(initial_X.reshape(-1,self.dim))
            initial_X = initial_X.reshape(self.num_bounds, -1, self.dim)
        train_y = []
        train_x = []
        for sample_points_per_bound in initial_X:
            x_per_bound = []
            y_per_bound = []
            for x in sample_points_per_bound:
                new_candidate_target = self.eval_func(x)
                # if self.check_exception(x, new_candidate_target):
                #     logger.info( "parameter {} caused floating point error {}".format(x, new_candidate_target))
                #     # if we detect error at initialization stage, we log the error and try again
                #     continue
                x_per_bound.append(x)
                y_per_bound.append(new_candidate_target)
            train_y.append(jnp.asarray(y_per_bound))
            train_x.append(jnp.asarray(x_per_bound))
        
        train_x = jnp.stack(train_x, axis=0)
        train_y = jnp.stack(train_y, axis=0)
        train_x = train_x.reshape(-1, self.dim)
        train_y = train_y.reshape(-1,1)
        if normalize:
            pt = MinMaxScaler()
            train_y = pt.fit_transform(train_y)
        self.dataset = gpx.Dataset(X=train_x, y=train_y)

    def initialize_model(self, params=None):
        self.GP = GPModel(self.dataset)
        print(self.GP.params)
        self.GP.fit()
        print(self.GP.params)
        if params is not None:
            self.GP.params = params

    def suggest_new_candidate(self, n_warmup=500, n_samples=5):
        # TODO: add sampling around best point.
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
        x_tries = self.random_sample(n_warmup)

        ys = jit(vmap(self.ucb))(x_tries)
        ys = jnp.nan_to_num(ys, nan=2.1219957915e-314)
        max_acq, max_indices = self.max_argmax(ys)
        x_max = jnp.take_along_axis(x_tries, max_indices,axis=1)
        
        # Explore the parameter space more throughly
        x_seeds = self.random_sample(n_samples)
        
        clamped_candidates = jnp.clip(x_seeds, self.lower_bounds, self.upper_bounds)

        to_minimize = lambda x: -self.ucb(x).sum()

        # Exponential decay of the learning rate.
        scheduler = ox.exponential_decay(
            init_value=1e-7, 
            transition_steps=1000,
            decay_rate=0.99)

        # Combining gradient transforms using `optax.chain`.
        optimizer = ox.chain(
            ox.clip_by_global_norm(5.0),  # Clip by the gradient by the global norm.
            ox.scale_by_adam(),  # Use the updates from adam.
            ox.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
            # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
            ox.scale(-1.0)
        )

        opt_state = optimizer.init(clamped_candidates)

        for i in range(100):
            X = jnp.clip(clamped_candidates, self.lower_bounds, self.upper_bounds)
            grads = jit(vmap(grad(to_minimize)))(X)

            updates, opt_state = optimizer.update(grads, opt_state)
            clamped_candidates = ox.apply_updates(clamped_candidates, updates)
            if jnp.isnan(clamped_candidates).sum() > 0:
                clamped_candidates = X
                break

        clamped_candidates = jnp.clip(clamped_candidates, self.lower_bounds, self.upper_bounds)
        batch_acquisition = jit(vmap(self.ucb))(clamped_candidates)
        batch_acquisition = jnp.nan_to_num(batch_acquisition, nan=2.1219957915e-314)

        best_acqs, best_candidate_indices = self.max_argmax(batch_acquisition)
        clamped_candidates = jnp.take_along_axis(clamped_candidates, best_candidate_indices, axis=1)

        condition = max_acq > best_acqs
        best_candidates = jnp.where(condition, x_max, clamped_candidates)
        return best_candidates

    def check_exception(self, param, val):
        # TODO: check exceptions in batch.
        # TODO: Flatten the the bounds + num_samples (B X N X D -> B*N X D) and run the function normally. Tested on all bound at once. Remove bounds that encounter exception.
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
        if jnp.isinf(val):
            if val < 0.0:
                self.results["min_inf"] += 1
                self.exception_induced_params["min_inf"].append(param)
                # self.save_trials_to_trigger(exp_name)
            else:
                self.results["max_inf"] += 1
                self.exception_induced_params["max_inf"].append(param)
                # self.save_trials_to_trigger(exp_name)
            return True

        # Subnormals
        if jnp.isfinite(val):
            if val > -2.22e-308 and val < 2.22e-308:
                if val != 0.0 and val != -0.0:
                    if val < 0.0:
                        self.results["min_under"] += 1
                        self.exception_induced_params["min_under"].append(param)
                    else:
                        self.results["max_under"] += 1
                        self.exception_induced_params["max_under"].append(param)
                    return True

        if jnp.isnan(val):
            self.results["nan"] += 1
            self.exception_induced_params["nan"].append(param)
            return True
        return False

    def random_sample(self, num_sample):
        key = jr.PRNGKey(246)
        key, subkey = jr.split(key)
        return jr.uniform(
            subkey, 
            shape=(self.num_bounds, num_sample, self.dim), 
            dtype=dtype, 
            minval=self.lower_bounds, 
            maxval=self.upper_bounds
            )

    def ucb(self, x, kappa=2.5):
        mean, std = self.GP.predict(x)
        return mean + kappa * std

    def max_argmax(self, array, keepdims=True, axis=-1):
        max_indices = array.argmax(axis=axis, keepdims=keepdims)
        max_indices = jnp.expand_dims(max_indices, axis=axis)
        max_values = array.max(axis=axis, keepdims=keepdims)
        max_values = jnp.expand_dims(max_values, axis=axis)
        return max_values, max_indices

    def train(self):
        # TODO: reimpliment fit function in mixed precision.
        print("Begin BO")
        start_fitting = time.time()
        for i in range(self.iteration):
            if i % self.batch_size == 0 and i != 0:
                old_state_dict = self.GP.params
                self.initialize_model(state_dict=old_state_dict)
                for param_name, param in self.GP.params:
                    if param.isnan():
                        # self.initialize_model(state_dict=old_state_dict)
                        print(param_name, param)
                        break
            new_candidates = self.suggest_new_candidate()
            new_targets = []
            # best_target = -1e+307
            # best_candidate = 0
            remove_bounds_indices = torch.zeros(self.lower_bounds.shape[0])
            for i, candidate in enumerate(new_candidates):
                new_candidate_target = self.eval_func(candidate.detach())
                new_candidate_target = torch.as_tensor([new_candidate_target], device=self.device, dtype=dtype)
                if self.check_exception(candidate, new_candidate_target.detach()):
                    logger.info(
                        "parameter {} caused floating point error {}".format(candidate, new_candidate_target.detach()))
                    remove_bounds_indices[i] = 1
                    continue
                # self.GP(candidate.unsqueeze(0))
                # if new_candidate_target > best_target:
                #     best_target = new_candidate_target
                #     best_candidate = candidate
                new_targets.append(new_candidate_target)
            new_targets = jnp.asarray(new_targets, dtype=dtype)[remove_bounds_indices==0]
            new_candidates = new_candidates[remove_bounds_indices==0]
            new_data = gpx.Dataset(new_candidates, new_candidate_target)
            self.dataset = self.dataset(new_data)
                #self.GP = self.GP.get_fantasy_model(candidate.unsqueeze(0), new_candidate_target)
            self.lower_bounds = self.lower_bounds[remove_bounds_indices==0]
            self.upper_bounds = self.upper_bounds[remove_bounds_indices==0]
            assert self.train_x.shape[0] == self.train_y.shape[0], f"shape mismatch, got {self.train_x.shape[0]} for training data but {self.train_y.shape[0]} for testing data"

        print("Fitting time: ", time.time() - start_fitting)

        # for type in self.error_types:
        #     self.exception_induced_params[type] = torch.cat(self.exception_induced_params[type]).flatten()
