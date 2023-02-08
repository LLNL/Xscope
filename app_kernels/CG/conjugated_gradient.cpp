#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <math.h>
using namespace std;

const double NEARZERO = 1.0e-10;       // interpretation of "zero"

using vec    = std::vector<double>;         // vector
using matrix = std::vector<vec>;            // matrix (=collection of (row) vectors)

// Prototypes
void print(const vec &a);
void write_to_file(const vec &a, ofstream &file);
vec matrixTimesVector( const matrix &A, const vec &V );
vec vectorCombination( double a, const vec &U, double b, const vec &V );
double innerProduct( const vec &U, const vec &V );
double vectorNorm( const vec &V );
vec conjugateGradientSolver( const matrix &A, const vec &B );
double kernel_1(double x0, double x1, double x2, double x3, 
  double x4, double x5, double x6, double x7, 
  double x8, double x9, double x10, double x11, 
  double x12, double x13, double x14, double x15,
  double y0, double y1, double y2, double y3);
// double kernel_wrapper_1(
//     double x0, double x1, double x2, double x3, 
//     double x4, double x5, double x6, double x7, 
//     double x8, double x9, double x10, double x11, 
//     double x12, double x13, double x14, double x15,
//     double y0, double y1, double y2, double y3);

//======================================================================

// write out the vector to file
void print(const vec &a) {
   for(int i=0; i < a.size(); i++)
      cout << a.at(i) << ' ';
}

//======================================================================

// write out the vector to file
void write_to_file(const vec &a, ofstream &file) {
   for(int i=0; i < a.size(); i++)
      file << a.at(i) << ' ';
    file << "\n";
}
//======================================================================

// Inner product of the matrix A with vector V returned as C (a vector)
vec matrixTimesVector( const matrix &A, const vec &V )     // Matrix times vector
{
  int n = A.size();
  vec C( n );
  for ( int i = 0; i < n; i++ ) C[i] = innerProduct( A[i], V );
  return C;
}


//======================================================================

// Returns the Linear combination of aU+bV as a vector W.
vec vectorCombination( double a, const vec &U, double b, const vec &V )        // Linear combination of vectors
{
  int n = U.size();
  vec W( n );
  for ( int j = 0; j < n; j++ ) W[j] = a * U[j] + b * V[j];
  return W;
}


//======================================================================

// Returns the inner product of vector U with V.
double innerProduct( const vec &U, const vec &V )          // Inner product of U and V
{
  return inner_product( U.begin(), U.end(), V.begin(), 0.0 );
}


//======================================================================

// Computes and returns the Euclidean/2-norm of the vector V.
double vectorNorm( const vec &V )                          // Vector norm
{
  return sqrt( innerProduct( V, V ) );
}

//======================================================================

// The conjugate gradient solving algorithm.
vec conjugateGradientSolver( const matrix &A, const vec &B )
{
  // Setting a tolerance level which will be used as a termination condition for this algorithm
  double TOLERANCE = 1.0e-10;

  // Number of vectors/rows in the matrix A.
  int n = A.size() * 10;

  // Initializing vector X which will be set to the solution by this algorithm.
  vec X( A.size(), 0.0 );
  vec R = B;
  vec P = R;
  int k = 0;

  while ( k < n )
  {
    vec Rold = R;                                         // Store previous residual
    vec AP = matrixTimesVector( A, P );

    //
    double alpha = innerProduct( R, R ) / max( innerProduct( P, AP ), NEARZERO );
    X = vectorCombination( 1.0, X, alpha, P );            // Next estimate of solution
    R = vectorCombination( 1.0, R, -alpha, AP );          // Residual

    if ( vectorNorm( R ) < TOLERANCE ) break;             // Convergence test

    double beta = innerProduct( R, R ) / max( innerProduct( Rold, Rold ), NEARZERO );
    P = vectorCombination( 1.0, R, beta, P );             // Next gradient
    k++;
  }

  return X;
}

double kernel_1(
  double x0, double x1, double x2, double x3, 
  double x4, double x5, double x6, double x7, 
  double x8, double x9, double x10, double x11, 
  double x12, double x13, double x14, double x15,
  double y0, double y1, double y2, double y3) {
    matrix A = {{x0, x1, x2, x3},
     {x4, x5, x6, x7},
     {x8, x9, x10, x11},
     {x12, x13, x14, x15}};
    vec B = {y0, y1, y2, y3};
    vec predicted_x = conjugateGradientSolver(A, B);
    vec predicted_b = matrixTimesVector(A, predicted_x);
    ofstream error_file;
    error_file.open ("CG_fp_errors.txt", std::ios_base::app);
    // if (predicted_b != B){
    //   error_file << "CG miscalculated X because of theses inputs: \n";
    //   error_file << "The vector elements in B are : \n";
    //   print(B, error_file);
    //   error_file << "The vector elements in predicted B are : \n";
    //   print(predicted_b, error_file);
    //   error_file << "The matrix elements in A are : \n";
    //   error_file << x0 << " " << x1 << " " << x2 << " " << x3 << " " 
    //   << x4 << " " << x5 << " " << x6 << " " << x7 << " " 
    //   << x8 << " " << x9 << " " << x10 << " " << x11 << " " 
    //   << x12 << " " << x13 << " " << x14 << " " << x15 << "\n";
    // }
    for ( int j = 0; j < predicted_x.size(); j++ ){
      double value = predicted_x.at(j);
      if (!isnormal(value) && value!=0){
          error_file << "Exception found in CG output: \n";
          write_to_file(predicted_x, error_file);
          print(predicted_x);
          return value;
          };
    }
    error_file.close();

    // for(double i : X.size()) {
    //   if (!std::isnormal(i)){
    //       return i;
    //   };
    // }

    double norm = vectorNorm(predicted_x);
    if (!isnormal(norm) && norm!=0){
          cout << "it is the vectorNorm that caused exception \n";
          };
    return norm;
}

extern "C" {
double kernel_wrapper_1(
    double x0, double x1, double x2, double x3, 
    double x4, double x5, double x6, double x7, 
    double x8, double x9, double x10, double x11, 
    double x12, double x13, double x14, double x15,
    double y0, double y1, double y2, double y3) {
  double res = kernel_1(x0, x1, x2, x3, 
                    x4, x5, x6, x7, 
                    x8, x9, x10, x11, 
                    x12, x13, x14, x15,
                    y0, y1, y2, y3);
  return res;
  }
 }
