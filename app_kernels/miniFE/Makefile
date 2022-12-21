

all:
	nvcc -arch=sm_35 cuda_code_minife.cu -o cuda_code_minife.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
