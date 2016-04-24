CC=cilk++
CFLAGS=-O3 
#-I $(HOME)/cilk/include/cilk++

all:	FW_Aloop FW_Bloop FW_Cloop FW_Dloop aloop bloop cloop dloop rec2 rec3 rec4

FW_Aloop:
	$(CC) $(CFLAGS) -o FW_Aloop FW_Aloop.cilk -lcilkutil
FW_Bloop:
	$(CC) $(CFLAGS) -o FW_Bloop FW_Bloop.cilk -lcilkutil
FW_Cloop:
	$(CC) $(CFLAGS) -o FW_Cloop FW_Cloop.cilk -lcilkutil
FW_Dloop:
	$(CC) $(CFLAGS) -o FW_Dloop FW_Dloop.cilk -lcilkutil
aloop:
	nvcc -arch=compute_35 -code=sm_35 Aloop.cu -o aloop
bloop:
	nvcc -arch=compute_35 -code=sm_35 Bloop.cu -o bloop
cloop:
	nvcc -arch=compute_35 -code=sm_35 Cloop.cu -o cloop
dloop:
	nvcc -arch=compute_35 -code=sm_35 Dloop.cu -o dloop
rec2:
	nvcc -arch=compute_35 -code=sm_35 rec2.cu -o rec2
rec3:
	nvcc -arch=compute_35 -code=sm_35 rec3.cu -o rec3  
rec4:
	nvcc -arch=compute_35 -code=sm_35 rec4.cu -o rec4  

clean:
	rm FW_Aloop FW_Bloop FW_Cloop FW_Dloop aloop bloop cloop dloop rec2 rec3 rec4
