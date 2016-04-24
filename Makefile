CC=cilk++
CFLAGS=-O3 
#-I $(HOME)/cilk/include/cilk++
all:	FW_aloop FW_bloop FW_cloop FW_dloop Aloop Bloop Cloop Dloop Cloop2
FW_aloop:
	$(CC) $(CFLAGS) -o FW_Aloop FW_Aloop.cilk -lcilkutil
FW_bloop:
	$(CC) $(CFLAGS) -o FW_Bloop FW_Bloop.cilk -lcilkutil
FW_cloop:
	$(CC) $(CFLAGS) -o FW_Cloop FW_Cloop.cilk -lcilkutil
FW_dloop:
	$(CC) $(CFLAGS) -o FW_Dloop FW_Dloop.cilk -lcilkutil
Aloop:
	nvcc -arch=compute_35 -code=sm_35 Aloop.cu -o aloop
Dloop:
	nvcc -arch=compute_35 -code=sm_35 Dloop.cu -o dloop
Bloop:
	nvcc -arch=compute_35 -code=sm_35 Bloop.cu -o bloop
Cloop:
	nvcc -arch=compute_35 -code=sm_35 Cloop.cu -o cloop
Cloop2:
	nvcc -arch=compute_35 -code=sm_35 Cloop2.cu -o cloop2
clean:
	rm -rf FW_Aloop FW_Bloop FW_Cloop FW_Dloop aloop bloop dloop cloop cloop2 core*
