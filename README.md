# XScope

XScope finds inputs that trigger floating-point exceptions, such as NaN (not a number) and 
infinity, in CUDA functions using Bayesian optimization (BO). XScope assumes that the CUDA 
functions are a black box, i.e., the source code is not available. It searches the input space 
using several methods to guide BO into extreme cases. When an input is found to trigger an 
exception in the target CUDA function, the input is shown to the user.

For more details (and to cite this work), please see the SC22 paper:
```
Ignacio Laguna, Ganesh Gopalakrishnan, “Finding Inputs that Trigger Floating-Point 
Exceptions in GPUs via Bayesian Optimization”. The International Conference for 
High Performance Computing, Networking, Storage and Analysis (SC22), Dallas, TX, 
USA, Nov 13-18, 2022.
```

To reproduce the SC22 paper results, see `reproducing_SC22_results.md`.  

### Driver
The main driver is `xscope`, which takes several options:
```
$ ./xscope.py -h
usage: xscope.py [-h] [-a AF] [-n NUMBER_SAMPLING] [-r RANGE_SPLITTING] 
[-s SAMPLES] [--random_sampling] [--random_sampling_unb] [-c] FUNCTION_TO_TEST

Xscope tool

positional arguments:
  FUNCTION_TO_TEST      Function to test (file or shared library .so)

optional arguments:
  -h, --help            show this help message and exit
  -a AF, --af AF        Acquisition function: ei, ucb, pi
  -n NUMBER_SAMPLING, --number-sampling NUMBER_SAMPLING
                        Number sampling method: fp, exp
  -r RANGE_SPLITTING, --range-splitting RANGE_SPLITTING
                        Range splitting method: whole, two, many
  -s SAMPLES, --samples SAMPLES
                        Number of BO samples (default: 30)
  --random_sampling     Use random sampling
  --random_sampling_unb
                        Use random sampling unbounded
  -c, --clean           Remove temporal directories (begin with _tmp_)
```


### License
XScope is distributed under the terms of the MPI license. 
All new contributions must be made under the MIT license.

See LICENSE and NOTICE for details.

LLNL-CODE-836653
