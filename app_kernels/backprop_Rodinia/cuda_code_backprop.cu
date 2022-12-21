
__device__
double  squash(double x0) {
  //float m;
  //x = -x;
  //m = 1 + x + x*x/2 + x*x*x/6 + x*x*x*x/24 + x*x*x*x*x/120;
  //return(1.0 / (1.0 + m));
  return (1.0 / (1.0 + exp(-x0)));
}

__global__ void kernel_1(
  double x0, double *ret) {
   *ret = squash(x0);
}

extern "C" {
double kernel_wrapper_1(double x0) {
  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0,dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }


