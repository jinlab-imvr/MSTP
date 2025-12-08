# MSTP: Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/jinlab-imvr/MSTP)
[![GitHub](https://img.shields.io/github/stars/jinlab-imvr/MSTP?style=social)](https://github.com/jinlab-imvr/MSTP)

**[🌐 Project Page](https://github.com/jinlab-imvr/MSTP) · [📄 Paper](https://arxiv.org/abs/2509.17429)**

</div>

---

### Installation

> Tested with **2 × NVIDIA H200 Tensor Core GPUs**

```bash
git clone https://github.com/jinlab-imvr/MSTP.git
cd MSTP/LLaMA-Factory

# Create and activate the environment
conda create -n mstp python=3.10 -y
conda activate mstp

# Install core dependencies
pip install wheel
pip install -e ".[torch,metrics]" --no-build-isolation

# (Optional) Choose transformers version by model family
# For Qwen2.5-VL series pretrained models:
pip install transformers==4.51
# For InternVL3 and gemma-3 series pretrained models:
pip install transformers==4.52

# Additional requirements
pip install -r requirements.txt
```

---

### Dataset

The dataset provided in the paper can be downloaded for verification.

- We use **8 video frames** from the [GraSP](https://github.com/BCV-Uniandes/GraSP) dataset for **training** and **4 video frames** for **testing**.
- Please run [`make_augment_all.py`](make_augment_all.py) to perform data augmentation.
- If you want to obtain the processed labels for the MSTP task used in the paper:
  - Fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfVumHC4jRMs9IMPbFYjr_mI8_k6ZHCwZCi4a_aER3rq2qCfA/viewform?usp=header) to obtain the download link.
  - After download, extract the compressed file to [`LLaMA-Factory/data/`](LLaMA-Factory/data/).

If you want to customize the dataset, please refer to the [data instructions](LLaMA-Factory/data/README.md).

---

### Weights of Visual Generation Module

Download the pretrained **visual generation model weights** (formerly SD weights) we provide, and place them in the `pretrained` directory:

- [`ioky/SD3.5_large`](https://huggingface.co/ioky/SD3.5_large)
- [`ioky/SD3.5_medium`](https://huggingface.co/ioky/SD3.5_medium)

---

### LoRA Weights of Decision-making Module

Download the LoRA weights of the pretrained **decision-making models** (formerly VL models) we trained, and place them in the `LoRA` directory:

- [`ioky/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/ioky/Qwen2.5-VL-7B-Instruct)
- [`ioky/InternVL3-8B-hf`](https://huggingface.co/ioky/InternVL3-8B-hf)
- [`ioky/gemma-3-4b-it`](https://huggingface.co/ioky/gemma-3-4b-it)

---

### Inference of Temporal Prediction via Incremental Generation

```bash
cd MSTP/LLaMA-Factory

# Use Qwen2.5-VL-7B-Instruct as the decision-making model
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 \
    --data_dir dir_to_dataset --sd_model large --mode test \
    --model_name Qwen2.5-VL-7B-Instruct

# Use gemma-3-4b-it as the decision-making model
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 \
    --data_dir dir_to_dataset --sd_model large --mode test \
    --model_name gemma-3-4b-it

# Use InternVL3-8B-hf as the decision-making model
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 \
    --data_dir dir_to_dataset --sd_model large --mode test \
    --model_name InternVL3-8B-hf
```

---

### Training of Visual Generation Module

To fine-tune the **SD3.5-based visual generation model**, please refer to the official  
[Stable Diffusion 3.5 fine-tuning guide](https://stabilityai.notion.site/Stable-Diffusion-3-5-fine-tuning-guide-11a61cdcd1968027a15bdbd7c40be8c6).

---

### Training of Decision-making Module

This project uses **LoRA** to train the **decision-making models** (formerly VL models).

```bash
cd MSTP/LLaMA-Factory
DISABLE_VERSION_CHECK=1 llamafactory-cli train \
    examples/train_lora/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```
```bash
DISABLE_VERSION_CHECK=1 llamafactory-cli export \
    examples/merge_lora/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

---

### Validation of MSTP

Generate decision-making model results in batches:

```bash
DISABLE_VERSION_CHECK=1 llamafactory-cli train \
    examples/predict/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

---

### Citing Our Work

If you find this code useful for your research, please cite:

```bibtex
@misc{zeng2025multiscaletemporalpredictionincremental,
      title        = {Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration},
      author       = {Zhitao Zeng and Guojian Yuan and Junyuan Mao and Yuxuan Wang and Xiaoshuang Jia and Yueming Jin},
      year         = {2025},
      eprint       = {2509.17429},
      archivePrefix= {arXiv},
      primaryClass = {cs.CV},
      url          = {https://arxiv.org/abs/2509.17429},
}
```
