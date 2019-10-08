#!/bin/bash

#SBATCH --job-name=transformdata
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=99:00:00

module purge
module load anaconda3/4.3.1
source activate py3.5
module load ffmpeg/intel/3.2.2
module load sox/intel/14.4.2 
module load portaudio19/gnu/20140130
#module load jupyter-kernels/py2.7
#module load jupyter-kernels/py3.5

python /home/ci411/SpokenWeb/data_transform.py


