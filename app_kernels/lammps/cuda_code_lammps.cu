
__device__ double phi_coeff[10];

__device__
double get_cg_p_corr(double vavg, double vCG, int N_basis, int N_mol)
{
  double correction = 0.0;
  for (int i = 1; i <= N_basis; ++i)
    correction -= phi_coeff[i-1] * ( N_mol * i / vavg ) *
      pow( ( 1 / vavg ) * ( vCG - vavg ),i-1);
  return correction;
}

__global__ void kernel_1(
  double x0,double x1,
  double *ret) {

  for (int i=0; i < 10; ++i)
    phi_coeff[i] = i;

  *ret =  get_cg_p_corr(x0, x1, 10, 10);
}

extern "C" {
double kernel_wrapper_1(double x0,double x1) {
  
  /* Init rand*/
  srand(22);
  double x2 = 1.0/((double)rand());
  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0,x1,dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }


