#!/bin/bash
#SBATCH -A cin_extern02
#SBATCH -p m100_usr_prod
#SBATCH --time 00:20:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2        # 2 gpus per node out of 4
#SBATCH --job-name=ddp

module load anaconda/2020.11
module load cuda/11.0
module load cmake
module load profile/deeplrn

source "/cineca/prod/opt/tools/anaconda/2020.11/none/etc/profile.d/conda.sh"

MYENV=$CINECA_SCRATCH/phd-ai/hpdl-env

conda activate $MYENV

cd $CINECA_SCRATCH/phd-ai/hpdl

srun python -W ignore -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 code/ddp.py --num_epochs 5 --model_dir data

conda deactivate
