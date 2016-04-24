The folder contains the following relevant files:

Aloop.cu - GPU implementation of Aloop-FW
Bloop.cu - GPU implementation of Bloop-FW
Cloop.cu - GPU implementation of Cloop-FW
Dloop.cu - GPU implementation of Dloop-FW

FW_Aloop.cilk - CPU implementation of Aloop-FW
FW_Bloop.cilk - CPU implementation of Bloop-FW
FW_Cloop.cilk - CPU implementation of Cloop-FW
FW_Dloop.cilk - CPU implementation of Dloop-FW

rec2.cu - GPU implementation of algorithm for question b)
rec3.cu	- GPU implementation of	algorithm for question c)
rec4.cu	- GPU implementation of	algorithm for question d)

findMaxSize.cu - Code to determine value of ng
Makefile - To generate executables
job.sh   - To submit jobs to be run. This file has code to run cilk program for iterative version
gjob.sh  - To submit jobs to be run. This file has code to run on GPU. Here the queue used is gpudev


