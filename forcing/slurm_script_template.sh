#!/bin/bash
#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=d1_i1_elasticities
#SBATCH --account=acclimat
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err
#SBATCH --workdir='++dir++'
#SBATCH --cpus-per-task=16
#SBATCH --export=ALL,OMP_PROC_BIND=FALSE
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
/home/robinmid/repos/acclimate/acclimate/build/acclimate ++dir++/settings_sandy.yml
