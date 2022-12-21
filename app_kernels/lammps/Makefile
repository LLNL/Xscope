

all:
	nvcc -arch=sm_35 cuda_code_lammps.cu -o cuda_code_lammps.cu.so -shared -Xcompiler -fPIC

clean:
	rm -rf *.so
