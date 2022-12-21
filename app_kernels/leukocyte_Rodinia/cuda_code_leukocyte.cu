
#define PI 3.141592653589793116
#define ONE_OVER_PI (1.0 / PI)

__device__ double heaviside(double x0) {
  return (atan(x0) * ONE_OVER_PI) + 0.5;
}

__global__ void kernel_1(
  double x0, double *ret) {
   *ret = heaviside(x0);
}

extern "C" {
double kernel_wrapper_1(double x0) {
  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0, dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }


