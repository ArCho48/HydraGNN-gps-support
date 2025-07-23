#!/bin/bash
#SBATCH -A LRN070
#SBATCH -J HydraGNN
#SBATCH -o job-%j.out
#SBATCH -e job-%j.out
#SBATCH -t 2:00:00
#SBATCH -p batch
##SBATCH -q debug
#SBATCH -N 10

# Frontier User Guide: https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
source /lustre/orion/lrn070/world-shared/mlupopa/module-to-load-frontier-rocm631.sh
source activate /lustre/orion/lrn070/world-shared/mlupopa/hydragnn_rocm631_venv

export PYTHONPATH=/lustre/orion/lrn070/world-shared/mlupopa/ADIOS_ROCm631_venv/adios2-install/lib/python3.11/site-packages/:$PYTHONPATH

#export LD_PRELOAD=/lib64/libgcc_s.so.1:/usr/lib64/libstdc++.so.6

export MIOPEN_DISABLE_CACHE=1

# setup hostfile
HOSTS=.hosts-job$SLURM_JOB_ID
HOSTFILE=hostfile.txt
srun hostname > $HOSTS
sed 's/$/ slots=8/' $HOSTS > $HOSTFILE

# Configuration 
export NNODES=$SLURM_JOB_NUM_NODES # e.g., 10 total nodes
export NNODES_PER_TRIAL=1
export NUM_CONCURRENT_TRIALS=$(( $NNODES / $NNODES_PER_TRIAL ))
export NTOTGPUS=$(( $NNODES * 8 )) # e.g., 80 total GPUs
export NGPUS_PER_TRIAL=$(( 8 * $NNODES_PER_TRIAL )) # e.g., 8 GPUs per training
export NTOT_DEEPHYPER_RANKS=$(( $NTOTGPUS / $NGPUS_PER_TRIAL )) # e.g., 10 total DH ranks
export OMP_NUM_THREADS=4 # e.g., 8 threads per rank
[ $NTOTGPUS -ne $(($NGPUS_PER_TRIAL*$NUM_CONCURRENT_TRIALS)) ] && echo "ERROR!!" 

#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# DeepHyper variables
export DEEPHYPER_LOG_DIR="deephyper-experiment"-$SLURM_JOB_ID 
#mkdir -p $DEEPHYPER_LOG_DIR
export DEEPHYPER_DB_HOST=$HOST

# Safe sleep to let everything start
sleep 5

echo "Doing something"

export PYTHONPATH=$PWD:$PYTHONPATH

cd examples/niaid

# Launch DeepHyper (1 rank per node, NTOT_DEEPHYPER_RANKS <= NNODES here)
python niaid_deephyper_parallel_main.py 
