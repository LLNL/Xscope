.PHONY: all
all: 
	nvcc -Ieigen-3.4.0/ -arch=sm_35 eigen_matrix.cu -o eigen_matrix.cu.so -shared -Xcompiler -fPIC 

.PHONY: clean
clean:
	rm -rf *.so