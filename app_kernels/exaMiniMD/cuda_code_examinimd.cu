
__device__ int switch_flag = 1;

__device__
static const double MY_PI  = 3.14159265358979323846; // pi

__device__
double compute_sfac(double r, double rcut, double rmin0)
{
  if (switch_flag == 0) return 1.0;
  if (switch_flag == 1) {
    if(r <= rmin0) return 1.0;
    else if(r > rcut) return 0.0;
    else {
      double rcutfac = MY_PI / (rcut - rmin0);
      return 0.5 * (cos((r - rmin0) * rcutfac) + 1.0);
    }
  }
  return 0.0;
}

__global__ void kernel_1(
  double x0,double x1,double x2,double *ret) {
   *ret = compute_sfac(x0,x1,x2);
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


