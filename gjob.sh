#!/bin/bash
#SBATCH -J Test        # Job Name
#SBATCH -o Test.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -n 16         # Total number of  tasks requested
#SBATCH -p gpudev  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 04:00:00     # Run time (hh:mm:ss) - 1.5 hours

./aloop 512 > aloop_512
./bloop 512 > bloop_512
./cloop 512 > cloop_512
./dloop 512 > dloop_512

./rec2 512 > rec2_512
./rec3 512 > rec3_512
./rec4 512 > rec4_512
