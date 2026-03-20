#!/bin/bash
srun --account=ufdatastudios --nodes=1 --gpus=1 --time=03:00:00 --mem=64G --cpus-per-task=8 --partition=hpg-b200 --pty bash
