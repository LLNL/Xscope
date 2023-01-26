#include <stdlib.h>
#include <stdio.h>
#include <Eigen/Dense>
__device__
double min_eigen_values(Eigen::Matrix4d A)
{
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigensolver(A);
  return eigensolver.eigenvalues().minCoeff();
}

__device__
double matrix_determinant(Eigen::Matrix4d A)
{
  return A.determinant();
}

__global__ void kernel_1(
  double x0, double x1, double x2, double x3, 
  double x4, double x5, double x6, double x7, 
  double x8, double x9, double x10, double x11, 
  double x12, double x13, double x14, double x15, 
  double *ret) {
    Eigen::Matrix4d A;
    A << x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15;
    *ret = min_eigen_values(A);
}

extern "C" {
double kernel_wrapper_1(
double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7, 
double x8, double x9, double x10, double x11, double x12, double x13, double x14, double x15) {
  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_1<<<1,1>>>(x0, x1, x2, x3, 
                    x4, x5, x6, x7, 
                    x8, x9, x10, x11, 
                    x12, x13, x14, x15,
                    dev_p);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }
