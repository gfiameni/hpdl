#!/bin/bash
#SBATCH --nodes=2               # number of nodes
#SBATCH --ntasks-per-node=4     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of threads per task
#SBATCH --threads-per-core=1    # number of threads per core
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time 0:30:00          # format: HH:MM:SS
#SBATCH --exclusive
#SBATCH --mem 0
#SBATCH --job-name=mddp3
#SBATCH -o %x.o%j
#SBATCH -e %x.e%j
#SBATCH --gres-flags=enforce-binding
#SBATCH --overcommit
#SBATCH --parsable
#SBATCH -A 
#SBATCH -p m100_usr_prod

module load autoload profile/deeplrn cineca-ai

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12340

#srun python mddp.py
srun -l --mpi=none --ntasks 8 --ntasks-per-node 4  python code/ddp_multinode.py
