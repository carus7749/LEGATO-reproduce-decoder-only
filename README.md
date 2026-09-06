<div align="center">

# LEGATO Optical Music Recognition — Reproduction & Debugging

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-End--to--End%20Inference-success)
![License](https://img.shields.io/badge/License-MIT-green)

**Reproduction and debugging of an open-source multimodal Optical Music Recognition pipeline**

</div>

---

## Overview

This repository documents my reproduction and debugging of [LEGATO](https://github.com/guang-yng/legato), an open-source end-to-end Optical Music Recognition (OMR) model that converts score images into symbolic ABC notation.

The project progressed from a **decoder-only validation workflow** under constrained server connectivity to a complete **score-image → ABC inference pipeline** after deploying the vision backbone and resolving multiple implementation and multimodal compatibility issues.

### Final Status

- ✅ LEGATO decoder and main-model loading validated
- ✅ ~23 GB Llama-3.2-11B-Vision backbone deployed to the remote Linux server
- ✅ Processor, tensor-conversion, kwargs, and generation compatibility issues resolved
- ✅ End-to-end score-image → ABC inference completed
- ✅ Reproducible environment, scripts, logs, test data, and outputs packaged

> **Scope note:** This project reproduces and debugs the original LEGATO implementation. I did **not** develop the LEGATO architecture itself. The current repository demonstrates successful end-to-end inference; it does not claim reproduction of the paper's benchmark OMR accuracy.

---

## My Contribution

My work focused on reproducing the open-source system and making the inference pipeline operational:

- configured the remote Linux environment with Conda, PyTorch, and Hugging Face Transformers;
- diagnosed and fixed three implementation issues in `modeling_legato.py`;
- developed a decoder-only validation workflow when the vision backbone could not initially be downloaded to the server;
- downloaded the ~23 GB Llama-3.2-11B-Vision model locally with resumable downloads and transferred it to the server via `scp`;
- resolved processor initialization, NumPy-to-Tensor conversion, missing-kwargs, and unsupported-generation-argument errors;
- completed end-to-end **score image → ABC notation** inference;
- packaged repaired code, environment specifications, scripts, logs, test inputs, outputs, and documentation for reproducibility.

---

## Architecture

LEGATO follows a multimodal encoder-decoder workflow:

```text
Score Image
    ↓
Llama-3.2-11B-Vision
    ↓
Visual Features
    ↓
LEGATO Decoder
    ↓
Autoregressive Generation
    ↓
ABC Notation
```

The reproduction was completed incrementally so that failures in model loading, vision processing, and generation could be isolated and debugged separately.

---

## Reproduction Journey

```text
Original LEGATO repository
        ↓
Environment setup
        ↓
Implementation debugging
        ↓
Vision model unavailable on remote server
        ↓
Decoder-only validation
        ↓
Local download of ~23 GB vision backbone
        ↓
SCP transfer to remote Linux server
        ↓
Processor / tensor / kwargs debugging
        ↓
Generation compatibility debugging
        ↓
Score image → LEGATO → ABC notation
        ↓
End-to-end inference completed
```

---

# Stage 1 — Decoder-Only Validation

## Environment

The first stage was validated on:

- Ubuntu 20.04
- Python 3.12
- PyTorch 2.4.1
- Transformers 4.46.3
- CUDA-enabled GPU environment

The repository provides:

- `requirements.txt`
- `environment/conda_env_export.yaml`
- `scripts/setup_env.sh`

for environment reconstruction.

---

## Initial Implementation Fixes

Three issues were identified in `modeling_legato.py`:

| Issue | Problem | Fix |
|---|---|---|
| Vision-model attribute naming | `self.model.vision_model` was inconsistent with the class structure | Changed to `self.vision_model` |
| Missing fallback branch | `self.vision_model` could remain undefined when the encoder was skipped | Added `else: self.vision_model = None` |
| Encoder-loading restriction | `from_pretrained` prevented `load_pretrained_encoder=False` | Removed the conflicting restriction |

These changes allowed the main LEGATO model to load without requiring the vision backbone.

---

## Why Decoder-Only First?

The remote server could not directly access Hugging Face, while the Llama-3.2-11B-Vision backbone was approximately 23 GB.

Instead of blocking the reproduction at this point, I isolated the decoder-side pipeline and created `test_generate.py` to:

1. load the LEGATO tokenizer;
2. load the repaired main model;
3. bypass the visual encoder;
4. run autoregressive generation;
5. verify that ABC-formatted text could be produced.

This stage was a **sanity check for model loading and decoder generation**, not an evaluation of OMR recognition quality.

---

# Stage 2 — Full Vision Model Deployment

## Challenge

Full LEGATO inference requires the Llama-3.2-11B-Vision backbone, but the remote server could not directly download the model from Hugging Face.

## Solution

The model was downloaded in a local Windows environment using Hugging Face tools with resumable downloads.

During unstable network transfers, the download was resumed until all five `.safetensors` shards were obtained.

The complete model was then transferred to the server:

```text
Local machine
      ↓
Llama-3.2-11B-Vision (~23 GB)
      ↓
scp
      ↓
Remote Linux server
      ↓
~/llama-vision/
```

File sizes were checked after transfer to verify that the model shards were complete.

---

# Stage 3 — End-to-End Inference Debugging

Running the original `inference.py` exposed additional compatibility issues.

## Issue 1 — Processor Initialization

```text
TypeError:
MllamaProcessor.__init__()
missing 1 required positional argument: 'tokenizer'
```

**Cause**

`LegatoProcessor.__init__` did not explicitly pass the tokenizer required by the parent processor.

**Fix**

Updated the processor initialization to explicitly accept and forward the tokenizer.

---

## Issue 2 — NumPy / PyTorch Type Mismatch

```text
AttributeError:
'numpy.ndarray' object has no attribute 'to'
```

**Cause**

Some processor outputs were NumPy arrays, while the inference code attempted to call `.to(device)` directly.

**Fix**

Added type checking and converted NumPy arrays to PyTorch tensors before moving them to the target device.

---

## Issue 3 — Missing `common_kwargs`

```text
KeyError: 'common_kwargs'
```

**Cause**

The merged kwargs dictionary did not always contain this key.

**Fix**

Replaced direct indexing with a safe default:

```python
kwargs.get("common_kwargs", {})
```

---

## Issue 4 — Unsupported Generation Argument

```text
ValueError:
The following model_kwargs are not used by the model:
['use_model_defaults']
```

**Cause**

The generation call included an argument not accepted by the loaded model.

**Fix**

Removed `use_model_defaults` from the generation call.

---

# Results

After the vision backbone was deployed and the compatibility issues were resolved, the full inference command completed successfully:

```bash
conda activate legato
cd ~/legato

PYTHONPATH=. python scripts/inference.py \
  --model_path ../legato-model \
  --image_path test_images/simple.png
```

The model successfully:

```text
loaded the score image
        ↓
processed visual features
        ↓
loaded the LEGATO generation pipeline
        ↓
generated symbolic output
        ↓
saved JSON containing abc_transcription
```

Example generated ABC output:

```abc
AgAgA/B/G//2A/B/G/
I:linebreak $
K:C
V:1 treble
V:1
G2 g4 G2 d4 ...
```

The successful `abc_transcription` output demonstrates that the complete image-to-symbolic-notation inference path can run end to end.

> This result validates pipeline execution rather than benchmark recognition accuracy. Formal comparison against the paper using metrics such as TEDn or OMR-NED remains future work.

---

# Repository Structure

```text
LEGATO-reproduction/
├── README.md
├── report.md
├── code/
│   ├── modeling_legato.py
│   ├── inference.py
│   └── test_generate.py
├── config/
│   └── config.json
├── test_data/
├── outputs/
│   └── generated_abc.txt
├── environment/
│   └── conda_env_export.yaml
├── requirements.txt
└── scripts/
    ├── fix_modeling.py
    ├── setup_env.sh
    └── run_inference.sh
```

---

# Setup

## Option 1 — pip

```bash
pip install -r requirements.txt
```

## Option 2 — Conda

```bash
conda env create -f environment/conda_env_export.yaml
conda activate legato
```

The repository also includes:

```bash
bash scripts/setup_env.sh
```

for automated environment setup.

---

# Model Preparation

The LEGATO main model can be obtained from Hugging Face:

```bash
huggingface-cli login

huggingface-cli download \
  guangyangmusic/legato \
  --local-dir ./legato-model
```

Full OMR inference additionally requires the Llama-3.2-11B-Vision backbone.

Because the vision model is large and may be difficult to download directly from restricted servers, a practical workflow is:

```text
Download locally
      ↓
Verify model shards
      ↓
Transfer to server
      ↓
Configure local model path
      ↓
Run inference
```

---

# Limitations & Future Work

The current reproduction demonstrates successful end-to-end inference, but several extensions remain:

- evaluate recognition quality on a public benchmark using **TEDn / OMR-NED**;
- test additional real-world score images and analyze recognition errors;
- investigate remaining model-loading `MISSING` warnings caused by structural differences between LEGATO and the vision backbone;
- tune generation parameters such as `temperature` and `repetition_penalty`;
- explore fine-tuning on symbolic-music datasets such as PDMX-Synth;
- optionally expose the inference pipeline through a lightweight Gradio or FastAPI interface.

---

# Acknowledgements

This project is based on the original open-source [LEGATO repository](https://github.com/guang-yng/legato).

All credit for the original LEGATO architecture and research belongs to its authors. This repository documents my independent reproduction, debugging, deployment, and inference-validation work.
