#!/bin/bash

#SBATCH --job-name=pescador
#SBATCH --nodes=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=99:00:00

module purge
module load anaconda3/4.3.1
source activate py3.5
cd /home/ci411/SpokenWeb/Speaker-Diarization/
python train_pescador.py


