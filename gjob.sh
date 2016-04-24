#!/bin/bash
#SBATCH -J Test        # Job Name
#SBATCH -o Test.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -n 16         # Total number of  tasks requested
#SBATCH -p gpudev  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 04:00:00     # Run time (hh:mm:ss) - 1.5 hours

#export CILK_NWORKERS=32

#rm Test*

#./FW_Cloop 16 > 16_output
#./FW_Cloop 32 > 32_output
#./FW_Cloop 64 > 64_output
#./FW_Cloop 128 > 128_output
#./FW_Cloop 256 > 256_output
#./FW_Cloop 512 > 512_output
#./FW_Cloop 1024 > 1024_output
#./FW_Cloop 2048 > 2048_output

./rec3 4096> 4096_output
#./rec3 2048 > 2048_output
#./FW_Cloop 16384 > 16384_output
#./aloop 32768 > 32768_output
#./bloop 16384 > test16384b_output
#./aloop 8192 > test8192a_output
#./aloop 16384 > test16384a_output



#./FW_Bloop $1 > fw_bloop_output
#./FW_Cloop $1 > fw_cloop_output
#./FW_Dloop $1 > fw_dloop_output
