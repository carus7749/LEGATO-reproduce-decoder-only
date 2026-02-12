#!/bin/bash
# LEGATO 简化推理一键运行脚本（跳过视觉编码器）

cd ~/legato || { echo "❌ 未找到 legato 目录"; exit 1; }

# 确保环境激活
source ~/miniconda3/bin/activate legato

# 运行简化测试
echo "===== 开始推理 ====="
PYTHONPATH=. python test_generate.py