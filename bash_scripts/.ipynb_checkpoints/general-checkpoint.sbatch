#!/bin/bash
#SBATCH --account=ufdatastudios
#SBATCH --job-name=general
#SBATCH --output=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/general_%j.out
#SBATCH --error=/orange/ufdatastudios/dj.brinkley/predictions/bash_scripts/logs/general_%j.err
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --mail-user=dj.brinkley@ufl.edu
#SBATCH --mail-type=BEGIN,END,FAIL
