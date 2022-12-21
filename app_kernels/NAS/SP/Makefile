

all:
	nvcc -arch=sm_35 cuda_code_sp.cu -o cuda_code_sp.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
