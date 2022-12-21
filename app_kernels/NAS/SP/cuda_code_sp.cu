
__device__ double ce[13][6];
__device__ double dtemp[5];

__device__ static double exact_solution_gpu_device(const double xi, const double eta, const double zeta){
 //using namespace constants_device;
  for(int m=0; m<5; m++){
    dtemp[m]=ce[0][m]+xi*
  (ce[1][m]+xi*
(ce[4][m]+xi*
 (ce[7][m]+xi*
 ce[10][m])))+eta*
(ce[2][m]+eta*
 (ce[5][m]+eta*
 (ce[8][m]+eta*
 ce[11][m])))+zeta*
 (ce[3][m]+zeta*
 (ce[6][m]+zeta*
  (ce[9][m]+zeta*
  ce[12][m])));
  }

  return dtemp[0];
}

__global__ void kernel_1(
  double x0,double x1,double x2,
  double *ret) {

  for (int i=0; i < 13; ++i)
    for (int j=0; j < 6; ++j)
      ce[i][j] = 1.0/((double)i);

  *ret =  exact_solution_gpu_device(x0, x1, x2);
}

extern "C" {
double kernel_wrapper_1(double x0,double x1) {
  
  /* Init rand*/
  srand(22);
  double x2 = 1.0/((double)rand());
  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0,x1,x2,dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }


