#!/bin/bash

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize conda
source $HOME/miniconda3/bin/activate

# Create and activate environment
conda create -n diffusion python=3.10 -y
conda activate diffusion

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install required packages
pip install diffusers==0.25.0 \
    transformers \
    accelerate \
    safetensors \
    Pillow \
    requests \
    tqdm \
    datasets \
    wandb \
    bitsandbytes

