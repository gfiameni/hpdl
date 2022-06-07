#!/bin/bash
#SBATCH --nodes=2               # number of nodes
#SBATCH --ntasks-per-node=4     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --threads-per-core=1    # number of threads per core
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time 0:10:00          # format: HH:MM:SS
#SBATCH --exclusive

#SBATCH -A tra22_Nvaitc

#SBATCH -p m100_sys_test
#SBATCH --qos=qos_test

module load autoload profile/deeplrn cineca-ai

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12340

#srun python mddp.py
mpirun python mddp.py
