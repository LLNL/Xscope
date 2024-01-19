from utils.init import *

class Input_bound():
    def __init__(self, num_input=1, input_type="fp", device = torch.device("cuda"), input_range = None, f_type=None, params_per_group=None) -> None:
        self.device = device
        self.input_range = input_range
        if self.input_range is None:
            self.input_range = [-max_normal, max_normal] if input_type=="fp" else [-307,307]
        self.num_ranges = self.input_range.ndim
        self.params_per_group = params_per_group
        if self.params_per_group == None:
            self.params_per_group = self.num_input
        self.ignore_params = None
        self.num_input = num_input
        if self.num_input > self.params_per_group:
            self.set_ignore_params()

    def generate_bounds(self, split, ignore_params = None):
        if self.num_ranges==1:
            b = self.generate_bounds_single_ranges(split)
        else:
            b = self.generate_bounds_multi_ranges(split, ignore_params)
        self.bounds = torch.from_numpy(b).to(dtype=dtype, device=self.device)
        self.bounds = self.bounds
        self.num_bounds, _, self.dim = self.bounds.shape
        self.bounds_set = torch.split(self.bounds, 512)
    
    def update_dim(self, bounds):
        self.current_bounds = bounds
        self.num_bounds, _, self.dim = bounds.shape

    def generate_bounds_single_ranges(self, split):
        upper_lim = self.input_range[1]
        lower_lim = self.input_range[0]
        small_bound = 1e-100
        assert upper_lim >= lower_lim, f"upper bound{upper_lim} must be greater than lower bound {lower_lim}"
        group_size = min(self.params_per_group, self.num_input)
        if split == 1:
            limits = np.linspace(lower_lim, upper_lim, 10000)
            limits_slice = []
            ranges = []
            for lim in limits:
                limits_slice.append(np.array(lim).repeat(group_size))
            
            for i in range(len(limits_slice)-1):
                one_bound = np.stack([limits_slice[i], limits_slice[i+1]])
                ranges.append(one_bound)
            # lower_bound = np.array(lower_lim).repeat(group_size)
            # upper_bound = np.array(upper_lim).repeat(group_size)
            # b_norm = np.stack([lower_bound, upper_bound])
            sub_lower_bound = np.array([-small_bound]).repeat(group_size)
            zero_bound = np.zeros(group_size)
            sub_upper_bound = np.array([small_bound]).repeat(group_size)
            b_sub_negative = np.stack([sub_lower_bound, zero_bound])
            b_sub_positive = np.stack([zero_bound, sub_upper_bound])
            #b_sub = np.stack([sub_lower_bound, sub_upper_bound])
            ranges.append(b_sub_negative)
            ranges.append(b_sub_positive)
            #ranges.append(b_sub)
            # b = np.stack((b_norm, b_sub))
            b = np.stack(ranges)
            # b = np.expand_dims(b_norm, axis=0)
            return b
        # if 0 == lower_lim or (lower_lim < small_bound and lower_lim>0):
        #     if upper_lim > small_bound:
        #         lower_range = np.array([lower_lim ,small_bound])
        #         upper_range = np.linspace(small_bound, upper_lim, split)[1:]
        #         limits = np.concatenate((lower_range, upper_range), axis=0)
        #     else:
        #         limits = np.linspace(lower_lim, upper_lim, split+1)
        # elif 0 == upper_lim or (upper_lim> -small_bound and upper_lim <0):
        #     if lower_lim < -small_bound:
        #         lower_range = np.linspace(lower_lim, -small_bound, split)[:-1]
        #         upper_range = np.array([-small_bound,upper_lim])
        #         limits = np.concatenate((lower_range, upper_range), axis=0)
        #     else:
        #         limits = np.linspace(lower_lim, upper_lim, split+1)
        # else:
        #     limits = np.linspace(lower_lim, upper_lim, split+1)
        limits = np.linspace(lower_lim, upper_lim, split+1)
        ranges = []
        for i in range(split):
            ranges.append([limits[i], limits[i+1]])
            # if split > 2:
            ranges.append([-small_bound, small_bound])
        ranges = np.array(ranges)
        # if split == 2:
        #     b = ranges.repeat(self.params_per_group, axis = 0).reshape((2,self.params_per_group,2)).transpose(0,2,1)
        # else:
        if self.num_input == 1:
            return np.expand_dims(ranges, axis = -1)
        param_ranges = [np.arange(split+1) for _ in range(self.params_per_group)]
        #param_ranges = [np.arange(4) for _ in range(self.params_per_group)]
        param_grids = np.meshgrid(*param_ranges, indexing='ij')
        param_combinations_indices = np.stack(param_grids, axis=-1).reshape(-1, self.params_per_group)
        b = np.stack([np.take(ranges, indices, axis=0).transpose(1,0) for indices in param_combinations_indices])
        return b

    def generate_bounds_multi_ranges(self, split, ignore_params):
        upper_lim = self.input_range[:, 1]
        lower_lim = self.input_range[:, 0]
        #Splitting into chunk of 3 if there are more than 3 inputs
        if self.num_input > self.params_per_group:
            upper_lim_ranges = []
            lower_lim_ranges = []
            for i in range(self.num_input):
                if i not in ignore_params:
                    upper_lim_ranges.append(upper_lim[i])
                    lower_lim_ranges.append(lower_lim[i])
                for i in range(self.params_per_group-len(upper_lim_ranges)):
                    upper_lim_ranges.append(upper_lim[i])
                    lower_lim_ranges.append(lower_lim[i])

            upper_lim = np.array(upper_lim_ranges)
            lower_lim = np.array(lower_lim_ranges)
        assert (upper_lim >= lower_lim).all(), f"upper bound{upper_lim} must be greater than lower bound {lower_lim}"
       
        if split == 1:
            lower_bound = np.array(lower_lim)
            upper_bound = np.array(upper_lim)
            b = np.stack([lower_bound, upper_bound])
            b = np.expand_dims(b, axis=0)
        else:
            limits = np.linspace(lower_lim, upper_lim, split+1).transpose(1,0)
            ranges = []
            for lim in limits:
                single_range = []
                for i in range(split):
                    single_range.append([lim[i], lim[i+1]])
                ranges.append(single_range)
            ranges = np.array(ranges)
            param_ranges = [np.arange(split+2) for _ in range(self.params_per_group)]
            param_grids = np.meshgrid(*param_ranges, indexing='ij')
            param_combinations_indices = np.stack(param_grids, axis=-1).reshape(-1, self.params_per_group)
            b = np.stack([np.take(ranges, indices, axis=1)[:,0,:].transpose(1,0) for indices in param_combinations_indices])
        return b

    def set_ignore_params(self):
        params_index = torch.arange(self.num_input)
        params_combinations = torch.split(params_index, self.params_per_group)
        self.ignore_params=[]
        for split in params_combinations:
            ignore = [index for index in params_index if index not in split]
            ignore = torch.stack(ignore)
            self.ignore_params.append(ignore)
    
    def bounds_sampler(self, num_sample, padding=False, float_sample = True):
        #we only use float sample if the bound contains 0 and float_sample == True
        lbs = self.current_bounds[:,0,:]
        ubs = self.current_bounds[:,1,:]
        if float_sample:
            samples = []
            for i in range(self.current_bounds.shape[0]):
                lb = lbs[i]
                ub = ubs[i]
                single_bound_float_samples = np.zeros((self.dim, num_sample))
                for j in range(self.dim):
                    float_samples = self.float_uniform_random_generator(num_sample, ub[j].item(), lb[j].item())
                    single_bound_float_samples[j] = float_samples  
                single_bound_float_samples = single_bound_float_samples.transpose(1,0)
                samples.append(single_bound_float_samples)
            samples = np.stack(samples, axis=0)
            samples = torch.from_numpy(samples).to(device=self.device, dtype=dtype)
        else:
            samples = (ubs.unsqueeze(1)- lbs.unsqueeze(1)) * torch.rand((self.num_bounds, num_sample, self.dim), dtype= torch.double, device=torch.device("cuda")) + lbs.unsqueeze(1)
        if padding:
            samples = self.add_padding(samples)
        return samples

    def float_uniform_random_generator(self, rsample, upper_bound, lower_bound = 0):
        if upper_bound > 0 and lower_bound >= 0:   
            u_binary_repr = np.binary_repr(np.float64(upper_bound).view(np.uint64), width=64)
            int_upper_bound = int(u_binary_repr,2)

            l_binary_repr = np.binary_repr(np.float64(lower_bound).view(np.uint64), width=64)
            int_lower_bound = int(l_binary_repr,2)

            int_sample = np.random.randint(low = int_lower_bound,high = int_upper_bound, size = rsample, dtype=np.int64)
            float_sample = int_sample.view(np.float64)
        elif upper_bound <0 and lower_bound< 0:
            u_binary_repr = np.binary_repr(np.float64(upper_bound).view(np.uint64), width=64)
            int_upper_bound = -int(''.join('1' if x == '0' else '0' for x in u_binary_repr), 2) + 1

            l_binary_repr = np.binary_repr(np.float64(lower_bound).view(np.uint64), width=64)
            int_lower_bound = -int(''.join('1' if x == '0' else '0' for x in l_binary_repr), 2) + 1

            int_sample = np.random.randint(low=int_upper_bound, high=int_lower_bound, size=rsample, dtype=np.int64)
            float_sample = int_sample.view(np.float64)

        elif upper_bound == 0.0 and lower_bound <0:
            lower_bound, upper_bound = upper_bound, -lower_bound
            u_binary_repr = np.binary_repr(np.float64(upper_bound).view(np.uint64), width=64)
            int_upper_bound = int(u_binary_repr,2)

            l_binary_repr = np.binary_repr(np.float64(lower_bound).view(np.uint64), width=64)
            int_lower_bound = int(l_binary_repr,2)

            int_sample = np.random.randint(low = int_lower_bound,high = int_upper_bound, size = rsample, dtype=np.int64)
            float_sample = -int_sample.view(np.float64)
        else:
            lower_sample = int(rsample/2)
            upper_sample = rsample - lower_sample
            lower_half = self.float_uniform_random_generator(lower_sample, upper_bound=0, lower_bound=lower_bound)
            upper_half = self.float_uniform_random_generator(upper_sample, upper_bound=upper_bound, lower_bound=0)
            float_sample = np.concatenate((lower_half, upper_half))
        return float_sample
