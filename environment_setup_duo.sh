#!/usr/bin/env bash

conda create -yn duo python=3.10
conda activate duo

conda install -y git
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
conda install -y nvidia::cuda-cudart-dev
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformers==4.45.2 accelerate sentencepiece datasets wandb zstandard matplotlib huggingface_hub==0.25.2
pip install tensor_parallel==2.0.0

pip install ninja packaging
pip install flash-attn --no-build-isolation

# LongBench evaluation
pip install seaborn rouge_score einops pandas

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3/

# Install DuoAttention
pip install -e .