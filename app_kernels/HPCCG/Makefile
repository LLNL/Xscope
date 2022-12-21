

all:
	nvcc -arch=sm_35 cuda_code_hpccg.cu -o cuda_code_hpccg.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
