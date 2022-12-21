

all:
	nvcc -arch=sm_35 cuda_code_backprop.cu -o cuda_code_backprop.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
