#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:59:00
#SBATCH --output=/home/j46lei/projects/rrg-dclausi/j46lei/CIFAR-Image-Transformation/output/%x_%j.log
#SBATCH --account=rrg-dclausi
#SBATCH --mail-user=junf137@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE

# Usage: sbatch train_infer.sh <venv_path>
if [ "$#" -ne 1 ]; then
    echo "---* Usage: sbatch ${0##*/} <venv_path>"
    exit 1
fi

# Purge all loaded modules
echo "---* Purging all loaded modules..."
module --force purge

# Load necessary modules
echo "---* Loading required modules..."
module load StdEnv gcc opencv/4.10.0
module load python/3.10.13

# Activate the virtual environment
source $1/bin/activate

echo "---* Running the python script..."
export WANDB_MODE=offline
python ./autoencoder.py
