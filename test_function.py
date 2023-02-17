import os
import ctypes
import numpy
import sys
sys.path.append("interactive-rate-tendons/")
from cpptendon import tension_residual, tendon


class TestFunction:
    def __init__(
        self, 
        num_input, 
        mode="fp", 
        input_ranges=[0.0,1.0],
        num_task = 1, 
        is_custom_func = False
        ):
        self.lib = None
        self.MU = 1.0
        self.smallest_subnormal = 4e-323
        self.num_input = num_input
        self.mode = mode
        self.ignore_params = []
        if input_ranges.ndim == 1:
            self.params_list = numpy.random.default_rng(seed=123).uniform(input_ranges[0],  input_ranges[1], self.num_input).astype(numpy.float64)
        else:
            self.params_list = numpy.random.default_rng(seed=123).uniform(input_ranges[:,0],  input_ranges[:,1]).astype(numpy.float64)
        self.params_list_c = [ctypes.c_double(x) for x in self.params_list]
        self.num_task = num_task
        self.is_custom_func = is_custom_func
        self.robot = tendon.TendonRobot.from_toml("robot_specs.toml")

    def set_kernel(self, CUDA_LIB):
        self.lib = CUDA_LIB
        script_dir = os.path.abspath(os.path.dirname(__file__))
        lib_path = os.path.join(script_dir, self.lib)
        self.E = ctypes.cdll.LoadLibrary(lib_path)
        if self.num_task == 1:
            self.E.kernel_wrapper_1.restype = ctypes.c_double
        else:
            self.E.kernel_wrapper_1.argtypes = [ctypes.c_double for _ in self.num_task]

    def set_fn_type(self, fn_type):
        self.fn_type = fn_type
    
    def set_ignore_params(self, ignore_params):
        self.ignore_params = ignore_params

    def call_GPU_kernel_1(self, x):
        res = self.E.kernel_wrapper_1(ctypes.c_double(x))
        return res

    def call_GPU_kernel_2(self, x):
        x0, x1 = x[0], x[1]
        res = self.E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1))
        return res

    def call_GPU_kernel_3(self, x):
        x0, x1, x2 = x[0], x[1], x[2]
        res = self.E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2))
        return res

    def call_GPU_kernel_4(self, x):
        if len(self.ignore_params) > 0:
            param_pointer = 0
            for i in range(self.num_input):
                if i not in self.ignore_params:
                    self.params_list_c[i] = ctypes.c_double(x[param_pointer])
                    param_pointer += 1
        else:
            self.params_list_c = [ctypes.c_double(param) for param in x]
        res = self.E.kernel_wrapper_1(*self.params_list_c)
        return res

    
    #This is for testing, remove when done
    # def call_custom_python_functions(self,x):
    #     if len(self.ignore_params) > 0:
    #         param_pointer = 0
    #         for i in range(self.num_input):
    #             if i not in self.ignore_params:
    #                 self.params_list[i] = ctypes.c_double(x[param_pointer])
    #                 param_pointer += 1
    #     else:
    #         self.params_list = cuda.as_cuda_array(x)
        
    #     @jit(target_backend='cuda')
    #     def python_function(x1,x2,x3,x4,x5,x6):
    #         return (x1*x4) * (-x1+x2+x3-x4+x5+x6) + (x2*x5) * (x1-x2+x3+x4-x5+x6) + (x3*x6) * (x1+x2-x3+x4+x5-x6) - x2*x3*x4 - x1*x3*x5 - x1*x2*x6 - x4*x5*x6

    #     return res
    def custom_test(self, x):
        if len(self.ignore_params) > 0:
            param_pointer = 0
            for i in range(self.num_input):
                if i not in self.ignore_params:
                    self.params_list[i] = x[param_pointer]
                    param_pointer += 1
        else:
            self.params_list = x
        tau = self.params_list[:self.num_input]
        s_start = self.params_list[-1]
        res = tension_residual(tau, s_start, self.robot)
        return res
    
    def eval(self, x0):
        if self.mode == "exp":
            x0 = numpy.power(10, x0)
        if self.is_custom_func:
            r = self.custom_test(x0)
        elif self.num_input == 1:
            r = self.call_GPU_kernel_1(x0)
        elif self.num_input == 2:
            r = self.call_GPU_kernel_2(x0)
        elif self.num_input == 3:
            r = self.call_GPU_kernel_3(x0)
        else:
            r = self.call_GPU_kernel_4(x0)
        if self.num_task == 1:
            if self.fn_type == "min_inf":
                return -r
            elif self.fn_type == "max_under":
                if r == 0.0 or r == -0.0:
                    return -self.MU
                elif r > -self.smallest_subnormal:
                    return -r
            elif self.fn_type == "min_under":
                r = -r
                if r == 0.0 or r == -0.0:
                    return -self.MU
                elif r > -self.smallest_subnormal:
                    return -r
        else:
            r = numpy.asarray(r, dtype = numpy.float64)
            if self.fn_type == "min_inf":
                return -r
            elif self.fn_type == "max_under" or self.fn_type == "min_under":
                if self.fn_type == "min_under":
                    r = -r
                if r.any() == 0.0 or r.any() == -0.0:
                    mask = numpy.full_like(r, -self.MU)
                    condition = numpy.equal(r, 0.0)
                    return numpy.where(condition, mask, r)
                elif r.any() > -self.smallest_subnormal:
                    condition = numpy.greater(r, -self.smallest_subnormal)
                    return numpy.where(condition, -r, r)
        return r

