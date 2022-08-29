import os
import ctypes
import numpy

class TestFunction:
    def __init__(self):
        self.lib = None
        self.MU = 1.0
        self.smallest_subnormal = 4e-323

    def set_kernel(self, CUDA_LIB):
        self.lib = CUDA_LIB
        script_dir = os.path.abspath(os.path.dirname(__file__))
        lib_path = os.path.join(script_dir, self.lib)
        self.E = ctypes.cdll.LoadLibrary(lib_path)
        self.E.kernel_wrapper_1.restype = ctypes.c_double

    def call_GPU_kernel_1(self, x):
        x0 = x[0]
        res = self.E.kernel_wrapper_1(ctypes.c_double(x0))
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
        x0, x1, x2, x3 = x[0], x[1], x[2], x[3]
        res = self.E.kernel_wrapper_1(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2), ctypes.c_double(x3))
        return res

    def function_to_optimize(self, x0, num_input, func_type="max_inf", mode="fp"):
        if mode == "exp":
            x0 = numpy.power(10, x0)
        if num_input == 1:
            r = self.call_GPU_kernel_1(x0)
        elif num_input == 2:
            r = self.call_GPU_kernel_2(x0)
        elif num_input == 3:
            r = self.call_GPU_kernel_3(x0)
        else:
            r = self.call_GPU_kernel_4(x0)

        if func_type == "min_inf":
            return -r
        elif func_type == "max_under":
            if r == 0.0 or r == -0.0:
                return -self.MU
            elif r > -self.smallest_subnormal:
                return -r
        elif func_type == "min_under":
            if r == 0.0 or r == -0.0:
                return self.MU
            elif r < self.smallest_subnormal:
                return -r
        return r

