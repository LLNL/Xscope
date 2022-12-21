

all:
	nvcc -arch=sm_35 cuda_code_lulesh.cu -o cuda_code_lulesh.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
