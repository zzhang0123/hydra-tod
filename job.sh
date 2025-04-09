#!/bin/bash
# Initialize conda first
source "$(conda info --base)/etc/profile.d/conda.sh"

export JOBLIB_START_METHOD="forkserver"
export OMP_NUM_THREADS=13   
export VECLIB_MAXIMUM_THREADS=13

conda activate TOD

nohup mpiexec -n 2 python sim_and_test_multi_TOD.py > output.log 2>&1 &
echo $! > pid.file  # Save the process ID