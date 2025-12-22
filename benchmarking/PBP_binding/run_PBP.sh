#!/bin/bash
#SBATCH --job-name=pbp_binding
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:rtx6000:1            # Request 1 RTX 6000 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # 16 CPU cores
#SBATCH --mem=128G                      # 128 GB RAM
#SBATCH --time=5-00:00:00               # 5 days time limit
#SBATCH --tmp=200G                      # request 200 GB fast local NVMe storage
#SBATCH --output=logs/gpu_job_%j.out
#SBATCH --error=logs/gpu_job_%j.err

# Print job environment
echo "==========================================="
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# Setup local scratch
SCRATCH_DIR="/scratch/ariane/PBP_binding_V2/"
mkdir -p "$SCRATCH_DIR"

# Load modules (example - adjust when modules are available)
# module load cuda/12.1
# module load python/3.11
# Activate env
module load miniforge/25.9.1
eval "$(conda shell.bash hook)"
conda activate filterpipeline

# Stage data to fast local storage
#echo "Copying dataset to local scratch..."
cp /mnt/nfs/vol8t/home/amora/code/Filterzyme/benchmarking/PBP_binding/PBP_data_formatted.csv $SCRATCH_DIR
cp run_PBP.py $SCRATCH_DIR
cd $SCRATCH_DIR

# Run filrerzyme
echo "Starting filterzyme ..."
python3 run_PBP.py

# Copy results back to persistent storage
echo "Copying results back to home directory..."
#mkdir -p $HOME/results/job_$SLURM_JOB_ID
cp -r $SCRATCH_DIR/filterzyme_output/ /mnt/nfs/vol8t/home/amora/code/Filterzyme/benchmarking/PBP_binding/

# Cleanup SCRATCH -- not for now since I'll keep using it
#cd $SLURM_SUBMIT_DIR
#rm -rf "$SCRATCH_DIR"

echo "==========================================="
echo "Job completed: $(date)"
echo "Results saved to: /mnt/nfs/vol8t/home/amora/code/Filterzyme/benchmarking/PBP_binding/"
echo "$HOME/results/job_$SLURM_JOB_ID"
echo "==========================================="