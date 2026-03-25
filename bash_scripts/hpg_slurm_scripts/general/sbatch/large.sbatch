#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=b200_big
#SBATCH --output=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/b200_big_%j.out
#SBATCH --error=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/b200_big_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=dj.brinkley@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-b200
