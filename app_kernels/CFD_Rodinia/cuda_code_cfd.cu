
#define GAMMA 1.4

//__device__ double compute_pressure(double density, double density_energy, double speed_sqd)
__device__ double compute_pressure(double x0, double x1, double x2)
{
  return (double(GAMMA)-double(1.0))*(x1 - double(0.5)*x0*x2);
}

__global__ void kernel_1(
  double x0,double x1,double x2, double *ret) {
   *ret = compute_pressure(x0,x1,x2);
}

extern "C" {
double kernel_wrapper_1(double x0,double x1,double x2) {
  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0,x1,x2, dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }


