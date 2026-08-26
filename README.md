<div align="center">

# LEGATO Reproduction – Decoder Only

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-%23EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.46.3-yellow)](https://huggingface.co/docs/transformers/index)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/carus7749/LEGATO-reproduce-decoder-only?style=social)](https://github.com/carus7749/LEGATO-reproduce-decoder-only/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/carus7749/LEGATO-reproduce-decoder-only)](https://github.com/carus7749/LEGATO-reproduce-decoder-only/commits/main)

</div>

This repository documents a lightweight reproduction and debugging workflow based on [LEGATO](https://github.com/guang-yng/legato), an end-to-end optical music recognition model.

The project focuses on fixing implementation issues and validating decoder-only ABC notation generation under limited compute and storage resources.

> **Scope:** This is not a full image-to-ABC OMR reproduction. The visual encoder is excluded, so the current output validates decoder-side model loading and autoregressive ABC generation rather than optical music recognition accuracy.

---

## 1. 项目概述

LEGATO 是一个端到端光学音乐识别（OMR）模型，架构如下：

```
乐谱图像 → [Llama 视觉编码器] → 图像特征 → [LEGATO 解码器] → ABC 记谱法
                ↑                           ↑
            （需单独下载）              （主模型，本包已配置）
```


**本指南旨在跳过视觉编码器**，直接测试解码器部分，验证：
- Python 环境与依赖是否正确安装
- 修复后的 LEGATO 代码能否成功加载主模型权重
- 模型能否正常生成文本（ABC 格式）

**预期输出**：一段 ABC 乐谱文本（音符随机，但格式正确）。

---

## 2. 环境准备

### 2.1 硬件与操作系统
- **推荐**：Linux (Ubuntu 20.04+) 或 WSL2，至少 16GB 内存，GPU（可选但推荐）
- **已验证环境**：Ubuntu 20.04.5 LTS, CUDA 11.8, 12GB+ GPU

### 2.2 一键环境安装脚本
本证据包提供了自动化脚本 `scripts/setup_env.sh`，它会：
- 安装 Miniconda（如未安装）
- 创建 Python 3.12 的 conda 环境 `legato`
- 安装核心依赖（PyTorch 2.4.1, Transformers, Accelerate, Pillow, HuggingFace Hub, Datasets）

**执行命令**：
```bash
cd /path/to/LEGATO_reproduction_20260212
bash scripts/setup_env.sh
```

> **注意**：脚本执行后会激活 `legato` 环境，若未自动激活，请手动运行：
> ```bash
> source ~/miniconda3/bin/activate legato
> ```

### 2.3 手动安装（备选）
如需手动安装，请依次执行：
```bash
conda create -n legato python=3.12 -y
conda activate legato
pip install torch==2.4.1 transformers==4.46.3 accelerate==1.0.1 pillow==10.0.0 huggingface_hub datasets
```

---

## 3. 代码获取与修复

### 3.1 原始代码
LEGATO 官方代码仓库：  
`https://github.com/guang-yng/legato`

**本证据包已包含修复后的核心文件**，位于 `code/` 目录下：
- `modeling_legato.py`（模型定义，已修复）
- `inference.py`（原始推理脚本，已添加 `load_pretrained_encoder=False`）
- `test_generate.py`（简化测试脚本，完全跳过视觉编码器和处理器）
**因官方 processor 有 bug（已报告），简化测试绕过此模块**

### 3.2 修复说明（重要）
原始代码存在三处关键 Bug，修复详情如下：

| 问题 | 位置 | 修复方法 |
|------|------|----------|
| **命名不一致** | `modeling_legato.py` __init__ | 将 `self.model.vision_model` 改为 `self.vision_model`，统一属性名 |
| **缺少 `else` 分支** | `modeling_legato.py` __init__ | 添加 `else: self.vision_model = None`，避免属性未定义 |
| **`from_pretrained` 冲突检查** | `modeling_legato.py` from_pretrained | 移除对 `load_pretrained_encoder=False` 的限制，允许跳过视觉编码器加载 |

**修复脚本** `scripts/fix_modeling.py` 可一键应用所有修复（若从官方代码重新开始）。

---

## 4. 模型准备

### 4.1 主模型（解码器）
本包已提供 LEGATO 主模型的配置文件 `config/config.json`，模型权重文件（`model.safetensors`，约 429MB）需自行从 Hugging Face 下载：

```bash
huggingface-cli login  # 使用你的 token
huggingface-cli download guangyangmusic/legato --local-dir ./legato-model
```

**已上传服务器**：本证据包测试时使用的模型路径为 `~/legato-model`，包含所有必需文件（`config.json`, `tokenizer.json`, `model.safetensors` 等）。

### 4.2 视觉编码器（Llama-3.2-11B-Vision）—— **当前跳过**
完整 OMR 需要加载该编码器，但由于其体积大（20-30GB）且服务器网络受限，本指南**临时跳过**。  
当前流程仅验证解码器权重加载与 ABC 自回归生成；由于未加载视觉编码器，不能用于评估端到端 OMR 或识别准确率。

如需完整复现，请参考第 6 节。

---

## 5. 运行推理（简化版）

### 5.1 一键推理脚本
在环境已激活、代码和模型就绪的前提下，执行：

```bash
cd /path/to/legato
bash /path/to/LEGATO_reproduction_20260212/scripts/run_inference.sh
```

该脚本将：
- 进入 `~/legato` 目录（请根据实际情况修改路径）
- 激活 `legato` 环境
- 运行 `test_generate.py` 并输出结果

### 5.2 手动运行
```bash
conda activate legato
cd ~/legato
PYTHONPATH=. python test_generate.py
```

### 5.3 预期输出
控制台将显示类似以下内容：

```
加载分词器...
加载模型...
LegatoModel LOAD REPORT ...（大量 MISSING 警告，可忽略）
使用设备: cuda
生成中...

=== 生成结果 ===
X:1
T:Test
M:4/4
L:1/8
K:C
 |$
V:1 treble
V:1
 x/ !slide!PMTue !slide!d2 x3/4 |$ !slide!d2 x3/4 |$ !slide!d2 x3/4 |$ !slide!d2 x3/4 | %5
 !slide!d2 x3/4 |$ !slide!d2 x3/4 |$ !slide!
```

**成功标志**：出现 ABC 格式的文本块，说明模型加载成功、生成逻辑正常。

### 5.4 常见错误与解决

| 错误 | 原因 | 解决方法 |
|------|------|----------|
| `conda: command not found` | conda 未初始化 | 执行 `source ~/miniconda3/bin/activate legato` 或重启 shell |
| `No module named 'torch'` | 环境未激活或依赖未安装 | 确认在 `legato` 环境中，并执行 `pip install torch` |
| `LegatoModel ... MISSING` | 视觉编码器未加载 | 在 decoder-only 测试中可忽略；不适用于端到端 OMR |
| `OSError: ... config.json` | 模型路径错误 | 修改 `test_generate.py` 中的 `model_path` 变量 |

---

## 6. 完整 OMR 复现（后续工作）

若需进行端到端的乐谱图像→ABC 识别，必须加载视觉编码器 Llama-3.2-11B-Vision。  
步骤如下：

### 6.1 下载视觉编码器
确保服务器可访问 Hugging Face，执行：

```bash
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.2-11B-Vision --local-dir ~/llama-vision
```

### 6.2 修改主模型配置
编辑 `legato-model/config.json`，将：

```json
"encoder_pretrained_model_name_or_path": null
```
改为：
```json
"encoder_pretrained_model_name_or_path": "/path/to/llama-vision"
```

### 6.3 使用原始推理脚本
```bash
PYTHONPATH=. python scripts/inference.py \
    --model_path ../legato-model \
    --image_path your_score.png
```

**此时将不再出现 MISSING 警告，模型将利用真实视觉特征生成 ABC。**

---

## 7. 附录

### 7.1 依赖文件
本仓库提供两种环境重建方式：
- `requirements.txt`：核心 Python 依赖，适合使用 pip 安装
- `environment/conda_env_export.yaml`：已验证环境的完整 Conda 配置
使用 pip 安装：
```bash
pip install -r requirements.txt
**重建环境**：
```bash
conda env create -f conda_env_export.yaml
conda activate legato
```
或使用 `pip` 安装依赖：
```bash
pip install -r pip_freeze.txt
```

### 7.2 文件结构说明
```
LEGATO_reproduction_20260212/
├── README.md                     # 本指南
├── report.md                    # 向负责人汇报的总结
├── code/                        # 修复后的核心代码
│   ├── modeling_legato.py
│   ├── inference.py
│   └── test_generate.py
├── config/                      # 主模型配置文件
│   └── config.json
├── test_data/                   # 测试图片（simple.png）
├── requirements.txt
├── outputs/
│   └── generated_abc.txt
├── environment/
│   └── conda_env_export.yaml
└── scripts/                     # 一键复现脚本
    ├── fix_modeling.py
    ├── setup_env.sh
    └── run_inference.sh
```

### 7.3 致谢
感谢 LEGATO 论文作者提供开源代码，本复现工作基于官方仓库完成。  

