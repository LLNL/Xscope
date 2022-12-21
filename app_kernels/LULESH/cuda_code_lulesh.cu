#include <stdlib.h>
#include <stdio.h>
__device__
double CalcElemVolume( const double x0, const double x1,
               const double x2, const double x3,
               const double x4, const double x5,
               const double x6, const double x7,
               const double y0, const double y1,
               const double y2, const double y3,
               const double y4, const double y5,
               const double y6, const double y7,
               const double z0, const double z1,
               const double z2, const double z3,
               const double z4, const double z5,
               const double z6, const double z7 )
{
  double twelveth = double(1.0)/double(12.0);

  double dx61 = x6 - x1;
  double dy61 = y6 - y1;
  double dz61 = z6 - z1;

  double dx70 = x7 - x0;
  double dy70 = y7 - y0;
  double dz70 = z7 - z0;

  double dx63 = x6 - x3;
  double dy63 = y6 - y3;
  double dz63 = z6 - z3;

  double dx20 = x2 - x0;
  double dy20 = y2 - y0;
  double dz20 = z2 - z0;

  double dx50 = x5 - x0;
  double dy50 = y5 - y0;
  double dz50 = z5 - z0;

  double dx64 = x6 - x4;
  double dy64 = y6 - y4;
  double dz64 = z6 - z4;

  double dx31 = x3 - x1;
  double dy31 = y3 - y1;
  double dz31 = z3 - z1;

  double dx72 = x7 - x2;
  double dy72 = y7 - y2;
  double dz72 = z7 - z2;

  double dx43 = x4 - x3;
  double dy43 = y4 - y3;
  double dz43 = z4 - z3;

  double dx57 = x5 - x7;
  double dy57 = y5 - y7;
  double dz57 = z5 - z7;

  double dx14 = x1 - x4;
  double dy14 = y1 - y4;
  double dz14 = z1 - z4;

  double dx25 = x2 - x5;
  double dy25 = y2 - y5;
  double dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
   ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

  // 11 + 3*14
  double volume =
    TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
       dy31 + dy72, dy63, dy20,
       dz31 + dz72, dz63, dz20) +
    TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
       dy43 + dy57, dy64, dy70,
       dz43 + dz57, dz64, dz70) +
    TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
       dy14 + dy25, dy61, dy50,
       dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

  volume *= twelveth;

  return volume ;
}

__global__ void kernel_1(
  double x0,double x1,double x2,
  double x3, double x4, double x5, double x6, double x7,
  double y0, double y1, double y2, double y3, double y4, double y5, double y6, double y7,
  double z0, double z1, double z2, double z3, double z4, double z5, double z6, double z7,
  double *ret) {
   *ret = CalcElemVolume(x0,x1,x2, x3,x4,x5,x6,x7,
          y0, y1, y2, y3, y4, y5, y6, y7,            
          z0, z1, z2, z3, z4, z5, z6, z7           
          );
}

extern "C" {
double kernel_wrapper_1(
double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7, 
double y0, double y1, double y2, double y3, double y4, double y5, double y6, double y7, 
double z0, double z1, double z2, double z3, double z4, double z5, double z6, double z7) {
  
  /* Init rand*/
  // srand(22);
  // double x0 = 1.0/((double)rand());
  // double x1 = 1.0/((double)rand());
  // double x2 = 1.0/((double)rand());
  // double x3 = 1.0/((double)rand());
  // double x4 = 1.0/((double)rand());
  // double x5 = 1.0/((double)rand());
  // double x6 = 1.0/((double)rand());
  // double x7 = 1.0/((double)rand());
  // double y0 = 1.0/((double)rand());
  // double y4 = 1.0/((double)rand());
  // double y5 = 1.0/((double)rand());
  // double y6 = 1.0/((double)rand());
  // double y7 = 1.0/((double)rand());
  // double z0 = 1.0/((double)rand());
  // double z1 = 1.0/((double)rand());
  // double z2 = 1.0/((double)rand());
  // double z3 = 1.0/((double)rand());
  // double z4 = 1.0/((double)rand());
  // double z5 = 1.0/((double)rand());
  // double z6 = 1.0/((double)rand());
  // double z7 = 1.0/((double)rand());

  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0,x1,x2,
                    x3, x4, x5, x6, x7,
                    y0, y1, y2, y3, y4, y5, y6, y7,
                    z0, z1, z2, z3, z4, z5, z6, z7,
                    dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }


