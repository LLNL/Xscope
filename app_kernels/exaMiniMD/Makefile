

all:
	nvcc -arch=sm_35 cuda_code_examinimd.cu -o cuda_code_examinimd.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
