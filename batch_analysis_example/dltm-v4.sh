#!/bin/bash
#SBATCH --account=m4935
#SBATCH --image=registry.nersc.gov/m4935/dlesym-physicsnemo:25.06
#SBATCH --module=gpu,nccl-plugin
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --job-name=dltm_v4_analysis
#SBATCH --output=/pscratch/sd/n/nacc/dlesym_toolbox/analysis_outputs/dltm_v4/slurm-%j.out
#SBATCH --error=/pscratch/sd/n/nacc/dlesym_toolbox/analysis_outputs/dltm_v4/slurm-%j.err

# Optional: be explicit about GPU binding
export SLURM_GPU_BIND=none

# Launch the analysis suite
shifter bash -c "cd /pscratch/sd/n/nacc/dlesym_toolbox && python analysis_suite.py /pscratch/sd/n/nacc/dlesym_toolbox/analysis_configs/dltm_v4.yaml"
