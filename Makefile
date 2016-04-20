all:
	 nvcc -arch=compute_35 -code=sm_35 Aloop.cu -o test
clean:
	rm code_output Test*
