#!/bin/bash
# quick.sbatch
#SBATCH --account=ufdatastudios
#SBATCH --job-name=quick
#SBATCH --output=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/quick_%j.out
#SBATCH --error=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/quick_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-user=dj.brinkley@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200
