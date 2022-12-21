

//int compute_residual(const int n, const double * const v1, 
//		     const double * const v2, double * const residual)
__device__
double compute_residual(const int n, const double * const v1, const double * const v2)
{
  double local_residual = 0.0;
  double residual; 
 
  for (int i=0; i<n; i++) {
    double diff = fabs(v1[i] - v2[i]);
    if (diff > local_residual) local_residual = diff;
  }
#ifdef USING_MPI
  // Use MPI's reduce function to collect all partial sums

  double global_residual = 0;
  
  MPI_Allreduce(&local_residual, &global_residual, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  residual = global_residual;
#else
  residual = local_residual;
#endif

  return residual;
}

__global__ void kernel_1(
  double x0, double x1, double *ret) {
   *ret = compute_residual(1, &x0, &x1);
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
