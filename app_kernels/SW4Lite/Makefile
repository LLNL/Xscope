

all:
	nvcc -arch=sm_35 cuda_code_sw4lite.cu -o cuda_code_sw4lite.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
