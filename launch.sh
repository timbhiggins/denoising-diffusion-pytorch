#!/bin/bash
#PBS -N TrainDiffusion
#PBS -A UCUB0097
#PBS -l walltime=12:00:00
#PBS -o NEWt_TrainDiffusion.out
#PBS -e NEWt_TrainDiffusion.out
#PBS -q main
#PBS -l select=1:ncpus=64:mem=470GB:ngpus=4 -l gpu_type=a100
#PBS -m a
#PBS -M timothy.higgins@colorado.edu

# qsub -I -q main -A UCUB0097 -l walltime=12:00:00 -l select=1:ncpus=32:mem=470GB:ngpus=4 -l gpu_type=a100
# qsub -I -q casper -A UCUB0097 -l walltime=12:00:00 -l select=1:ncpus=32:mem=470GB:ngpus=1 -l gpu_type=a100

#accelerate config
module load conda
conda activate /glade/work/timothyh/miniconda3/envs/LuRain
accelerate launch Run_Training.py --lead_time=6, --dim_mults=(1,2,4,8)
