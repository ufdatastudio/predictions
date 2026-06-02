#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=l4_small
#SBATCH --output=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/l4_small_%j.out
#SBATCH --error=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/l4_small_%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=dj.brinkley@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=hpg-l4
