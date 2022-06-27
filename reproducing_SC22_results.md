# How to Reproduce SC22 Results

## Setting up the Docker Image
We provide a docker image with all the requirements to reproduce the key result of the SC22 paper.

### Step 0: Obtain the Docker Image
First, make sure the NVIDIA Container Toolkit is installed. See [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

Pull the image:
```
$ docker pull hpccorrectness/xscope_image
```

### Step 1: Run the Image
 To run the docker image, use the following:

```
$ sudo docker run --runtime=nvidia --gpus all -it hpccorrectness/xscope_image /bin/bash
```

### Step 2:
When you are in the bash shell, move to the xscope directory and unzip the source code. The zip file with the code is password-protected. The password is 0987xW*1234

```
$ cd xscope/
$ unzip xscope_source.zip 
Archive:  xscope_source.zip
[xscope_source.zip] xscope_source/functions_to_test.txt password:
```

Now move to the source code directory:
```
$ cd xscope_source
$ ls
README.md  app_kernels  bo_analysis.py  functions_to_test.txt  random_fp_generator.py  xscope.py
```

## Running the Experiments
We provide easy-to-use scripts to run the key experiments of the paper. These include the results presented in Table IV, Figure 7, Figure 8, and Figure 9, and Table V.

Obtaining the results from Figure 6 and Table III is possible, but it requires a significant amount of time since they are statistics of running all the MATH function calls.

The main script is `xscope`, which takes several options:
```
$ ./xscope.py -h
usage: xscope.py [-h] [-a AF] [-n NUMBER_SAMPLING] [-r RANGE_SPLITTING] [-s SAMPLES] [--random_sampling] [--random_sampling_unb] [-c] FUNCTION_TO_TEST

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

#### Results of Table IV and Runs Results (Fig. 7 and 8)
The input file most be modified: `./function_signatures.txt`. Functions can be commented by Ô#Õ. Simply uncomment the functions that will be tested. For example, this will produce the results for the `acos` function with the `whole` method for range splitting, and `fp` method for sampling:

```
$ ./xscope.py -r whole ./functions_to_test.txt 
Creating dir: _tmp_455c6931835b_13
Running: nvcc  -arch=sm_35 _tmp_455c6931835b_13/cuda_code_acos.cu -o _tmp_455c6931835b_13/cuda_code_acos.cu.so -shared -Xcompiler -fPIC
*** Running BO on: _tmp_455c6931835b_13/cuda_code_acos.cu.so
-------------- Results --------------
cuda_code_acos.cu.so
	INF+: 0
	INF-: 0
	SUB-: 0
	SUB-: 0
	NaN : 4
	Runs: 3
```
To use other methods simply modify the options -r and -s. By defaults it will use 30 samples for BO (which is what the paper uses in these figures and Tables). Note that the `many` method can take a significant amount of time to finish.

#### Results of Figure 9 (Random sampling)
For the random sampling results in Fig 9, use any of the following commands (bounded and unbounded methods, respectively):
```
$ ./xscope.py --random_sampling ./function_signatures.txt
$ ./xscope.py --random_sampling_unb ./function_signatures.txt
```

#### Results of Table V
Here we reproduce the columns of table V, except for Params, Ops and Time. Params and Ops are calculated manually (by looking at the code). The Time column will vary with the machine and GPU being used. The rest of the column values are fully reproducible.
For the application kernel results in Table V, first build the kernels:

```
$ cd app_kernels/
$ make
$ cd ..
```


Then, modify the last section of the input file: `./function_signatures.txt` by uncommenting the kernel to test (e.g., sw4lite):
```
#SHARED_LIB:./app_kernels/LULESH/cuda_code_lulesh.cu.so, 3
SHARED_LIB:./app_kernels/SW4Lite/cuda_code_sw4lite.cu.so, 2
#SHARED_LIB:./app_kernels/NAS/common/cuda_code_randlc.cu.so, 2
```

Then run ./xscope:

```
$ ./xscope.py ./function_signatures.txt
```



