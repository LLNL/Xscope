// Atomatically generated - do not modify

#include <stdio.h>
#include <cmath>

/*
FUNCTION:acos (double)
FUNCTION:acosh (double)
FUNCTION:asin (double)
FUNCTION:asinh (double)
FUNCTION:atan (double)
FUNCTION:atan2 (double, double)
FUNCTION:atanh (double)
FUNCTION:cbrt (double)
FUNCTION:ceil (double)
FUNCTION:copysign (double, double)
FUNCTION:cos (double)
FUNCTION:cosh (double)
FUNCTION:cospi (double)
FUNCTION:cyl_bessel_i1 (double)
FUNCTION:erf (double)
FUNCTION:erfc (double)
FUNCTION:erfcinv (double)
FUNCTION:erfcx (double)
FUNCTION:erfinv (double)
FUNCTION:exp (double)
FUNCTION:exp10 (double)
FUNCTION:exp2 (double)
FUNCTION:expm1 (double)
FUNCTION:fabs (double)
FUNCTION:fdim (double, double)
FUNCTION:floor (double)
FUNCTION:fmax (double, double)
FUNCTION:fmin (double, double)
FUNCTION:fmod (double, double)
FUNCTION:hypot (double, double)
FUNCTION:j0 (double)
FUNCTION:j1 (double)
FUNCTION:lgamma (double)
FUNCTION:log (double)
FUNCTION:log10 (double)
FUNCTION:log1p (double)
FUNCTION:log2 (double)
FUNCTION:logb (double)
FUNCTION:max (double, double)
FUNCTION:min (double, double)
FUNCTION:nearbyint (double)
FUNCTION:nextafter (double, double)
FUNCTION:normcdf (double)
FUNCTION:normcdfinv (double)
FUNCTION:pow (double, double)
FUNCTION:rcbrt (double)
FUNCTION:remainder (double, double)
FUNCTION:rhypot (double, double)
FUNCTION:rint (double)
FUNCTION:round (double)
FUNCTION:rsqrt (double)
FUNCTION:sin (double)
FUNCTION:sinpi (double)
FUNCTION:tan (double)
FUNCTION:tanh (double)
FUNCTION:tgamma (double)
FUNCTION:trunc (double)
FUNCTION:y0 (double)
FUNCTION:y1 (double)
*/


__global__ void kernel_fun(double x0,double x1,double *ret,int i) {
  if (i==0) *ret = acos (x0);
  if (i==1) *ret = acosh (x0);
  if (i==2) *ret = asin (x0);
  if (i==3) *ret = asinh (x0);
  if (i==4) *ret = atan (x0);
  if (i==5) *ret = atan2 (x0, x1);
  if (i==6) *ret = atanh (x0);
  if (i==7) *ret = cbrt (x0);
  if (i==8) *ret = ceil (x0);
  if (i==9) *ret = copysign (x0, x1);
  if (i==10) *ret = cos (x0);
  if (i==11) *ret = cosh (x0);
  if (i==12) *ret = cospi (x0);
  if (i==13) *ret = cyl_bessel_i1 (x0);
  if (i==14) *ret = erf (x0);
  if (i==15) *ret = erfc (x0);
  if (i==16) *ret = erfcinv (x0);
  if (i==17) *ret = erfcx (x0);
  if (i==18) *ret = erfinv (x0);
  if (i==19) *ret = exp (x0);
  if (i==20) *ret = exp10 (x0);
  if (i==21) *ret = exp2 (x0);
  if (i==21) *ret = expm1 (x0);
  if (i==23) *ret = fabs (x0);
  if (i==24) *ret = fdim (x0, x1);
  if (i==25) *ret = floor (x0);
  if (i==26) *ret = fmax (x0, x1);
  if (i==27) *ret = fmin (x0, x1);
  if (i==28) *ret = fmod (x0, x1);
  if (i==29) *ret = hypot (x0, x1);
  if (i==30) *ret = j0 (x0);
  if (i==31) *ret = j1 (x0);
  if (i==32) *ret = lgamma (x0);
  if (i==33) *ret = log (x0);
  if (i==34) *ret = log10 (x0);
  if (i==35) *ret = log1p (x0);
  if (i==36) *ret = log2 (x0);
  if (i==37) *ret = logb (x0);
  if (i==38) *ret = max (x0, x1);
  if (i==39) *ret = min (x0, x1);
  if (i==40) *ret = nearbyint (x0);
  if (i==41) *ret = nextafter (x0, x1);
  if (i==42) *ret = normcdf (x0);
  if (i==43) *ret = normcdfinv (x0);
  if (i==44) *ret = pow (x0, x1);
  if (i==45) *ret = rcbrt (x0);
  if (i==46) *ret = remainder (x0, x1);
  if (i==47) *ret = rhypot (x0, x1);
  if (i==48) *ret = rint (x0);
  if (i==49) *ret = round (x0);
  if (i==50) *ret = rsqrt (x0);
  if (i==51) *ret = sin (x0);
  if (i==52) *ret = sinpi (x0);
  if (i==53) *ret = tan (x0);
  if (i==54) *ret = tanh (x0);
  if (i==55) *ret = tgamma (x0);
  if (i==56) *ret = trunc (x0);
  if (i==57) *ret = y0 (x0);
  if (i==58) *ret = y1 (x0);
}

extern "C" {
double kernel_wrapper(double x0,double x1,int i) {
  double *dev_p;
  cudaMalloc(&dev_p, sizeof(double));
  kernel_fun<<<1,1>>>(x0,x1,dev_p,i);
  double res;
  cudaMemcpy (&res, dev_p, sizeof(double), cudaMemcpyDeviceToHost);
  return res;
  }
 }

