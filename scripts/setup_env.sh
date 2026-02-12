#!/bin/bash
# LEGATO 环境一键安装脚本

set -e  # 出错即停

echo "=== 安装 Miniconda（如果未安装） ==="
if ! command -v conda &> /dev/null; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    exec bash
fi

echo "=== 创建 legato 环境 ==="
conda create -n legato python=3.12 -y

echo "=== 激活环境 ==="
source ~/miniconda3/bin/activate legato

echo "=== 安装核心依赖 ==="
pip install torch transformers accelerate pillow huggingface_hub datasets

echo "=== 环境准备完成 ==="
echo "请运行: conda activate legato"