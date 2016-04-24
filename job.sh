#!/bin/bash
#SBATCH -J Test        # Job Name
#SBATCH -o Test.o%j    # Output and error file name (%j expands to jobID)
#SBATCH -n 16         # Total number of  tasks requested
#SBATCH -p development  # Queue (partition) name -- normal, development, etc.
#SBATCH -t 02:00:00     # Run time (hh:mm:ss) - 1.5 hours

export CILK_NWORKERS=32

./FW_Aloop 512 > FW_Aloop_512
./FW_Bloop 512 > FW_Bloop_512
./FW_Cloop 512 > FW_Cloop_512
./FW_Dloop 512 > FW_Dloop_512

