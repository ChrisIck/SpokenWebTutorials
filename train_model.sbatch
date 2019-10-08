#!/bin/bash

#SBATCH --job-name=spd_train
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=99:00:00

module purge
module load anaconda3/4.3.1
source activate py3.5
module load ffmpeg/intel/3.2.2
module load sox/intel/14.4.2 
module load portaudio19/gnu/20140130
#module load jupyter-kernels/py2.7
#module load jupyter-kernels/py3.5

cd /home/ci411/SpokenWeb/Speaker-Diarization/ghostvlad
#python predict.py
#python generate_embeddings.py
cd ..
python train.py


