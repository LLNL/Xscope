

all:
	nvcc -arch=sm_35 cuda_code_cfd.cu -o cuda_code_cfd.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
