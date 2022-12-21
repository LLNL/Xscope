

__device__
double randlc(double *x, double a, double r23, double t23, double r46, double t46){
  double t1,t2,t3,t4,a1,a2,x1,x2,z;
  t1 = r23 * a;
  a1 = (int)t1;
  a2 = a - t23 * a1;
  t1 = r23 * (*x);
  x1 = (int)t1;
  x2 = (*x) - t23 * x1;
  t1 = a1 * x2 + a2 * x1;
  t2 = (int)(r23 * t1);
  z = t1 - t23 * t2;
  t3 = t23 * z + a2 * x2;
  t4 = (int)(r46 * t3);
  (*x) = t3 - t46 * t4;

  return (r46 * (*x));
}



__global__ void kernel_1(
  double x0,double x1,double x2,
  double x3, double x4, double x5, 
  double *ret) {

  *ret =  randlc(&x0, x1, x2, x3, x4, x5);
}

extern "C" {
double kernel_wrapper_1(double x0,double x1) {
  
  /* Init rand*/
  srand(22);
  double x2 = 1.0/((double)rand());
  double x3 = 1.0/((double)rand());
  double x4 = 1.0/((double)rand());
  double x5 = 1.0/((double)rand());
  double x6 = 1.0/((double)rand());
  double x7 = 1.0/((double)rand());
  double y0 = 1.0/((double)rand());
  double y1 = 1.0/((double)rand());
  double y2 = 1.0/((double)rand());
  double y3 = 1.0/((double)rand());
  double y4 = 1.0/((double)rand());
  double y5 = 1.0/((double)rand());
  double y6 = 1.0/((double)rand());
  double y7 = 1.0/((double)rand());
  double z0 = 1.0/((double)rand());
  double z1 = 1.0/((double)rand());
  double z2 = 1.0/((double)rand());
  double z3 = 1.0/((double)rand());
  double z4 = 1.0/((double)rand());
  double z5 = 1.0/((double)rand());
  double z6 = 1.0/((double)rand());
  double z7 = 1.0/((double)rand());

  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0,x1,x2,x3,x4,x5,dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }


