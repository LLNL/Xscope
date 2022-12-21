
__device__
double ddot( unsigned n , const double *x , const double *y)
{
  double val = 0 ;
  const double * const x_end = x + n ;
  for ( ; x < x_end ; ++x , ++y ) { 
    val += *x * *y ; 
  }
  return val ;
}

__global__ void kernel_1(
  double x0, double x1, double *ret) {
   *ret = ddot(1, &x0, &x1);
}

extern "C" {
double kernel_wrapper_1(double x0, double x1) {
  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0,x1,dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }
