

all:
	nvcc -arch=sm_35 cuda_code_randlc.cu -o cuda_code_randlc.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
