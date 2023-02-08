from utils.init import *

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
        group_size = min(3,num_input)

        #Splitting into chunk of 3 if there are more than 3 inputs
        if num_input > 3:
            self.set_ignore_params(num_input)

        if split == 1:
            lower_bound = torch.as_tensor(lower_lim, dtype=torch.float64).repeat(group_size)
            upper_bound = torch.as_tensor(upper_lim, dtype=torch.float64).repeat(group_size)
            b = torch.stack((lower_bound,upper_bound)).unsqueeze(dim=0).to(dtype=dtype, device=device)
        else:
            limits = torch.linspace(lower_lim,upper_lim,split+1,dtype=torch.float64)
            print(limits)
            # ranges = torch.empty(split,2)
            # for i in range(split):
            #     ranges[i] = torch.tensor([limits[i], limits[i+1]])
            ranges = torch.combinations(limits, r=2)
            if num_input == 1:
                b = ranges.unsqueeze(-1)
            else:
                combinations = torch.arange(ranges.shape[0])
                combinations_indices = torch.combinations(combinations,group_size, with_replacement=True)
                b = torch.stack([torch.index_select(ranges,dim=0, index=index) for index in combinations_indices]).transpose(1,2)
                
        return b.to(dtype=dtype, device=self.device)

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


