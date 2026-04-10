<div align="center">
  <p><b>Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration</b></p>
  <p><i>NeurIPS 2025 Poster</i></p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS_2025-Poster-8A2BE2?style=for-the-badge" alt="NeurIPS 2025 Poster" />
  <img src="https://img.shields.io/badge/🔬_Surgical_Scenes-blue?style=for-the-badge" alt="Surgical Scenes" />
  <img src="https://img.shields.io/badge/🎥_Multi--Scale-green?style=for-the-badge" alt="Multi-Scale" />
  <img src="https://img.shields.io/badge/🤖_Multi--Agent-orange?style=for-the-badge" alt="Multi-Agent" />
  <img src="https://img.shields.io/badge/🖼️_Incremental_Generation-purple?style=for-the-badge" alt="Incremental Generation" />
  <br><br>
  <a href="https://arxiv.org/abs/2509.17429"><img src="https://img.shields.io/badge/📄_arXiv-2509.17429-red?style=flat-square" alt="arXiv" /></a>
  <a href="https://github.com/jinlab-imvr/MSTP"><img src="https://img.shields.io/badge/🌐_Project-Page-green?style=flat-square" alt="Project Page" /></a>
  <a href="https://github.com/jinlab-imvr/MSTP"><img src="https://img.shields.io/github/stars/jinlab-imvr/MSTP?style=flat-square" alt="GitHub Stars" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square" alt="License" /></a>
</p>

<p align="center">
  Accurate temporal prediction is the bridge between comprehensive scene understanding and embodied AI.<br>
  We formalize the <b>Multi-Scale Temporal Prediction (MSTP)</b> task in surgical and general scenes<br>
  and propose <b>Incremental Generation</b> and <b>Multi-agent Collaboration</b> to tackle it.
</p>

---

## 📝 Contents

- [Installation](#-installation)
- [Dataset](#-dataset)
- [Model Weights](#-model-weights)
- [Inference](#-inference)
- [Training](#-training)
- [Validation](#-validation)
- [Citation](#-citing-our-work)

---

## 🔧 Installation

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

## 📦 Dataset

The dataset provided in the paper can be downloaded for verification.

- We use **8 video frames** from the [GraSP](https://github.com/BCV-Uniandes/GraSP) dataset for **training** and **4 video frames** for **testing**.
- Please run [`make_augment_all.py`](make_augment_all.py) to perform data augmentation.
- If you want to obtain the processed labels for the MSTP task used in the paper:
  - Fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSd1SgP6u5ZCbUKKWqTsaHXnp2-mnggtgz6txDFy5heziCbbsA/viewform?usp=publish-editor) to obtain the download link.
  - After download, extract the compressed file to [`LLaMA-Factory/data/`](LLaMA-Factory/data/).

> If you want to customize the dataset, please refer to the [data instructions](LLaMA-Factory/data/README.md).

---

## 📥 Model Weights

### Visual Generation Module

Download the pretrained **visual generation model weights** and place them in the `pretrained` directory:

| Model | HuggingFace |
| --- | --- |
| SD3.5 Large | [`ioky/SD3.5_large`](https://huggingface.co/ioky/SD3.5_large) |
| SD3.5 Medium | [`ioky/SD3.5_medium`](https://huggingface.co/ioky/SD3.5_medium) |

### Decision-making Module (LoRA Weights)

Download the LoRA weights of the pretrained **decision-making models** and place them in the `LoRA` directory:

| Model | HuggingFace |
| --- | --- |
| Qwen2.5-VL-7B-Instruct | [`ioky/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/ioky/Qwen2.5-VL-7B-Instruct) |
| InternVL3-8B-hf | [`ioky/InternVL3-8B-hf`](https://huggingface.co/ioky/InternVL3-8B-hf) |
| gemma-3-4b-it | [`ioky/gemma-3-4b-it`](https://huggingface.co/ioky/gemma-3-4b-it) |

---

## 🚀 Inference

**Temporal Prediction via Incremental Generation**

| Decision-making Model | Command |
| --- | --- |
| Qwen2.5-VL-7B-Instruct | `python TP_IG.py --cir 5 --time 1 --start 0 --end 200 --data_dir dir_to_dataset --sd_model large --mode test --model_name Qwen2.5-VL-7B-Instruct` |
| gemma-3-4b-it | `python TP_IG.py --cir 5 --time 1 --start 0 --end 200 --data_dir dir_to_dataset --sd_model large --mode test --model_name gemma-3-4b-it` |
| InternVL3-8B-hf | `python TP_IG.py --cir 5 --time 1 --start 0 --end 200 --data_dir dir_to_dataset --sd_model large --mode test --model_name InternVL3-8B-hf` |

---

## 🏋️ Training

### Visual Generation Module

To fine-tune the **SD3.5-based visual generation model**, please refer to the official [Stable Diffusion 3.5 fine-tuning guide](https://stabilityai.notion.site/Stable-Diffusion-3-5-fine-tuning-guide-11a61cdcd1968027a15bdbd7c40be8c6).

### Decision-making Module

This project uses **LoRA** to train the decision-making models.

**Step 1 — Train:**
```bash
llamafactory-cli train examples/train_lora/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

**Step 2 — Export (merge LoRA):**
```bash
llamafactory-cli export examples/merge_lora/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

---

## 📊 Validation

Generate decision-making model results in batches:

```bash
llamafactory-cli train examples/predict/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

---

## 📖 Citing Our Work

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
