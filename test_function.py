import os
import ctypes
import numpy

class TestFunction:
    def __init__(self, num_input, mode="fp"):
        self.lib = None
        self.MU = 1.0
        self.smallest_subnormal = 4e-323
        self.num_input = num_input
        self.mode = mode

    def set_kernel(self, CUDA_LIB):
        self.lib = CUDA_LIB
        script_dir = os.path.abspath(os.path.dirname(__file__))
        lib_path = os.path.join(script_dir, self.lib)
        self.E = ctypes.cdll.LoadLibrary(lib_path)
        self.E.kernel_wrapper_1.restype = ctypes.c_double

    def set_fn_type(self, fn_type):
        self.fn_type = fn_type

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
        x0, x1, x2, x3, x4, x5, x6, x7 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]
        y0, y1, y2, y3, y4, y5, y6, y7 = x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]
        z0, z1, z2, z3, z4, z5, z6, z7 = x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23]

        res = self.E.kernel_wrapper_1(
            ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_double(x2), ctypes.c_double(x3),
            ctypes.c_double(x4), ctypes.c_double(x5), ctypes.c_double(x6), ctypes.c_double(x7),
            ctypes.c_double(y0), ctypes.c_double(y1), ctypes.c_double(y2), ctypes.c_double(y3),
            ctypes.c_double(y4), ctypes.c_double(y5), ctypes.c_double(y6), ctypes.c_double(y7),
            ctypes.c_double(z0), ctypes.c_double(z1), ctypes.c_double(z2), ctypes.c_double(z3),
            ctypes.c_double(z4), ctypes.c_double(z5), ctypes.c_double(z6), ctypes.c_double(z7))
        return res

    def eval(self, x0):
        if self.mode == "exp":
            x0 = numpy.power(10, x0)
        if self.num_input == 1:
            r = self.call_GPU_kernel_1(x0)
        elif self.num_input == 2:
            r = self.call_GPU_kernel_2(x0)
        elif self.num_input == 3:
            r = self.call_GPU_kernel_3(x0)
        else:
            r = self.call_GPU_kernel_4(x0)

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
        return r

