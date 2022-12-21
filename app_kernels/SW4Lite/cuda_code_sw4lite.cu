
#define float_sw4 double

__device__
float_sw4 VerySmoothBump(float_sw4 freq, float_sw4 t)
{
  float_sw4 tmp;
  if (t*freq < 0)
    tmp = 0.0;
  else if (t*freq > 1)
    tmp = 0.0;
  else
    tmp = - 1024*pow(t*freq,10) + 5120*pow(t*freq,9) - 10240*pow(t*freq,8) + 10240*pow(t*freq,7) - 5120*pow(t*freq,6) + 1024*pow(t*freq,5);
  return tmp;
}
 
__device__
float_sw4 VerySmoothBump_t(float_sw4 freq, float_sw4 t)
{
  float_sw4 tmp;
  if (t*freq < 0)
    tmp = 0.0;
  else if (t*freq > 1)
    tmp = 0.0;
  else
     tmp = freq*( - 1024*10*pow(t*freq,9) + 5120*9*pow(t*freq,8) - 10240*8*pow(t*freq,7) + 10240*7*pow(t*freq,6) - 5120*6*pow(t*freq,5) + 1024*5*pow(t*freq,4));
  return tmp;
}

__device__
float_sw4 VerySmoothBump_om(float_sw4 freq, float_sw4 t)
{
  float_sw4 tmp;
  if (t*freq < 0)
    tmp = 0.0;
  else if (t*freq > 1)
    tmp = 0.0;
  else
     tmp = t*( - 1024*10*pow(t*freq,9) + 5120*9*pow(t*freq,8) - 10240*8*pow(t*freq,7) + 10240*7*pow(t*freq,6) - 5120*6*pow(t*freq,5) + 1024*5*pow(t*freq,4));
  return tmp;
}

__global__ void kernel_1(
  double x0, double x1, double *ret) {
   *ret = VerySmoothBump_om(x0,x1);
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
