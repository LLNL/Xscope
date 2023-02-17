from utils.init import *
from itertools import combinations_with_replacement

class Input_bound():
    def __init__(self, num_input=1, input_type="fp", device = torch.device("cuda"), input_range = None, f_type=None) -> None:
        self.device = device
        self.input_range = input_range
        if self.input_range is None:
            self.input_range = [-max_normal, max_normal] if input_type=="fp" else [-307,307]
        self.num_ranges = self.input_range.ndim
        self.ignore_params = None
        if num_input > 3:
            self.set_ignore_params(num_input)
        self.num_input = num_input

    def generate_bounds(self, split, ignore_params):
        if self.num_ranges == 1:
            upper_lim = self.input_range[1]
            lower_lim = self.input_range[0]
        else:
            upper_lim = self.input_range[:, 1]
            lower_lim = self.input_range[:, 0]
        assert (upper_lim >= lower_lim).all(), f"upper bound{upper_lim} must be greater than lower bound {lower_lim}"
        group_size = min(3,self.num_input)

        #Splitting into chunk of 3 if there are more than 3 inputs
        if self.num_input > 3:
            upper_lim_ranges = []
            lower_lim_ranges = []
            for i in range(self.num_input):
                if i not in ignore_params:
                    upper_lim_ranges.append(upper_lim[i])
                    lower_lim_ranges.append(lower_lim[i])
            upper_lim = np.array(upper_lim_ranges)
            lower_lim = np.array(lower_lim_ranges)
        if split == 1:
            lower_bound = torch.as_tensor(lower_lim, dtype=torch.float64)
            upper_bound = torch.as_tensor(upper_lim, dtype=torch.float64)
            if self.num_ranges == 1:
                lower_bound = lower_bound.repeat(group_size)
                upper_bound = upper_bound.repeat(group_size)
            b = torch.stack((lower_bound,upper_bound)).unsqueeze(dim=0).to(dtype=dtype, device=device)
        else:
            limits = np.linspace(lower_lim, upper_lim, split).transpose(1,0)
            ranges = []
            for lim in limits:
                single_range = []
                for i in range(split-1):
                    single_range.append([lim[i], lim[i+1]])
                ranges.append(single_range)
            ranges = np.array(ranges)
            if self.num_input == 1:
                b = ranges.unsqueeze(-1)
            else:
                combination_index = np.array(list(combinations_with_replacement(np.arange(split-1), 3)))
                all_combination = []
                for comb in combination_index:
                    all_combination.append(ranges[range(3),comb,:])
                b = np.array(all_combination).transpose(0,2,1)
        self.bounds = torch.tensor(b).to(dtype=dtype, device=self.device)
        self.num_bounds, _, self.dim = self.bounds.shape

    def set_ignore_params(self, num_input):
        params_index = torch.arange(num_input)
        params_combinations = torch.split(params_index, 3)
        self.ignore_params=[]
        for split in params_combinations:
            ignore = [index for index in params_index if index not in split]
            ignore = torch.stack(ignore)
            self.ignore_params.append(ignore)
    
    def bounds_sampler(self, num_sample, padding=False):
        lb = self.bounds[:,0,:].unsqueeze(1)
        ub = self.bounds[:,1,:].unsqueeze(1)
        num_bounds = lb.shape[0]
        sampler = torch.distributions.uniform.Uniform(lb, ub)
        samples = sampler.rsample((num_sample,)).to(dtype=dtype, device=self.device).view(num_bounds, num_sample, self.dim)
        if padding:
            samples = self.add_padding(samples)
        return samples

    def update_bound(self, new_bound):
        self.bounds = new_bound
        self.num_bounds, _, self.dim = self.bounds.shape
        self.ignore_params = []


