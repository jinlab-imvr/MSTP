# MSTP: Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/jinlab-imvr/MSTP)
[![GitHub](https://img.shields.io/github/stars/jinlab-imvr/MSTP?style=social)](https://github.com/jinlab-imvr/MSTP)

**[🌐 Project Page](https://github.com/jinlab-imvr/MSTP) · [📄 Paper](https://arxiv.org/abs/2509.17429)**

</div>

---

## Environment Setup

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

## Constructed Surgical Dataset

The dataset provided in the paper can be downloaded for verification.

- We use **8 video frames** from the [GraSP](https://github.com/BCV-Uniandes/GraSP) dataset for **training** and **4 video frames** for **testing**.
- Please run [`make_augment_all.py`](make_augment_all.py) to perform data augmentation.
- If you want to obtain the processed labels for the MSTP task used in the paper:
  - Fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfVumHC4jRMs9IMPbFYjr_mI8_k6ZHCwZCi4a_aER3rq2qCfA/viewform?usp=header) to obtain the download link.
  - After download, extract the compressed file to [`LLaMA-Factory/data/`](LLaMA-Factory/data/).

If you want to customize the dataset, please refer to the [data instructions](LLaMA-Factory/data/README.md).

---

## Download Pretrained Base SD Model

Select a Stable Diffusion (SD) model for incremental generation and download it to the `pretrained` directory:

- [`stabilityai/stable-diffusion-3.5-large`](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

---

## Download Pretrained SD Weights

Download the SD weights we provide (trained on our task) to the `pretrained` directory:

- [`ioky/SD3.5_large`](https://huggingface.co/ioky/SD3.5_large)
- [`ioky/SD3.5_medium`](https://huggingface.co/ioky/SD3.5_medium)

---

## Download Pretrained Base VL Model

Select a vision-language (VL) model for multi-scale temporal prediction and download it to the `pretrained` directory:

- [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [`OpenGVLab/InternVL3-8B-hf`](https://huggingface.co/OpenGVLab/InternVL3-8B-hf)
- [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)

---

## Download Pretrained LoRA Weights of VL Model

Download the LoRA weights of the VL models we trained to the `LoRA` directory:

- [`ioky/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/ioky/Qwen2.5-VL-7B-Instruct)
- [`ioky/InternVL3-8B-hf`](https://huggingface.co/ioky/InternVL3-8B-hf)
- [`ioky/gemma-3-4b-it`](https://huggingface.co/ioky/gemma-3-4b-it)

---

## Temporal Prediction via Incremental Generation

```bash
cd MSTP/LLaMA-Factory

# Use Qwen2.5-VL-7B-Instruct
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 \
    --data_dir dir_to_dataset --sd_model large --mode test \
    --model_name Qwen2.5-VL-7B-Instruct

# Use gemma-3-4b-it
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 \
    --data_dir dir_to_dataset --sd_model large --mode test \
    --model_name gemma-3-4b-it

# Use InternVL3-8B-hf
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 \
    --data_dir dir_to_dataset --sd_model large --mode test \
    --model_name InternVL3-8B-hf
```

---

## SD Model Training

To fine-tune the SD3.5 model, please refer to the official  
[Stable Diffusion 3.5 fine-tuning guide](https://stabilityai.notion.site/Stable-Diffusion-3-5-fine-tuning-guide-11a61cdcd1968027a15bdbd7c40be8c6).

---

## VL Model Training

This project uses **LoRA** for training.

```bash
cd MSTP/LLaMA-Factory
DISABLE_VERSION_CHECK=1 llamafactory-cli train \
    examples/train_lora/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

---

## VL Model Validation

Generate VL model results in batches:

```bash
DISABLE_VERSION_CHECK=1 llamafactory-cli train \
    examples/predict/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

---

## VL Model Merge

```bash
DISABLE_VERSION_CHECK=1 llamafactory-cli export \
    examples/merge_lora/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

---

## Citing MSTP

If you find this work is useful for your research, please cite:

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
