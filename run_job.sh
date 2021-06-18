#!/bin/bash

#SBATCH -J artsy
#SBATCH -o TitanV.txt
#SBATCH -e stderr.txt
#SBATCH -p gpu_p
#SBATCH --qos=gpu  
#SBATCH --gres=gpu:1
#SBATCH --nodelist=supergpu03pxe
#SBATCH -t 00:10:00
#SBATCH -c 2
#SBATCH --mem=1G # dont use more than 50% of GPU queue node memory unless you request entire node
#SBATCH --nice=0 # priority

source ~/.bashrc
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/lib64
export TFHUB_CACHE_DIR=./tmp
conda activate artsyml

python video_stream_benchmark_parallel.py
