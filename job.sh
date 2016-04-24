#!/bin/bash
#SBATCH -J Test        # Job Name
#SBATCH -o Test.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -n 16         # Total number of  tasks requested
#SBATCH -p development  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 02:00:00     # Run time (hh:mm:ss) - 1.5 hours

export CILK_NWORKERS=32

rm Test*

./FW_Aloop 16 > 16_output
./FW_Aloop 32 > 32_output
./FW_Aloop 64 > 64_output
./FW_Aloop 128 > 128_output
./FW_Aloop 256 > 256_output
./FW_Aloop 512 > 512_output
./FW_Aloop 1024 > 1024_output
./FW_Aloop 2048 > 2048_output

./FW_Aloop 4096 > 4096_output
./FW_Aloop 8192 > 8192_output
./FW_Aloop 16384 > 16384_output
#./aloop 32768 > 32768_output
#./bloop 16384 > test16384b_output
#./aloop 8192 > test8192a_output
#./aloop 16384 > test16384a_output



#./FW_Bloop $1 > fw_bloop_output
#./FW_Cloop $1 > fw_cloop_output
#./FW_Dloop $1 > fw_dloop_output
