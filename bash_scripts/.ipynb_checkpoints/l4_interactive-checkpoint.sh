#!/bin/bash
srun --account=ufdatastudios --nodes=1 --gpus=1 --time=08:00:00 --mem=24G --cpus-per-task=4 --partition=hpg-l4 --pty bash
