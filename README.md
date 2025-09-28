# MSTP
Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration
<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/jinlab-imvr/MSTP)
[![GitHub](https://img.shields.io/github/stars/jinlab-imvr/MSTP?style=social)](https://github.com/jinlab-imvr/MSTP)

**[🌐 Project Page](https://github.com/jinlab-imvr/MSTP) | [📄 Paper](https://arxiv.org/abs/2509.17429) **

</div>

## Environment setup
Tested with 2 * NVIDIA H200 Tensor Core GPUs

```bash
git clone https://github.com/jinlab-imvr/MSTP.git
cd MSTP/LLaMA-Factory
# Create and activate the environment
conda create -n mstp python=3.10 -y
conda activate mstp

# Install dependencies
pip install wheel
pip install -e ".[torch,metrics]" --no-build-isolation

# If you want to use Qwen2.5-VL series pretrained model, install transformers==4.51
pip install transformers==4.51
# If you want to use InternVL3 and gemma-3 series pretrained model, install transformers==4.52
pip install transformers==4.52

pip install -r requirements.txt
```
## Constructed Surgical Dataset
The dataset provided in the article can be downloaded for verification purposes.

We use 8 video frames from the [GraSP](https://github.com/BCV-Uniandes/GraSP) dataset for training and 4 video frames for testing.
Please run the code [make_augment_all.py](make_augment_all.py) to perform data augmentation. If you want to obtain the processed labels of MSTP task used in the paper, please fill out this [form](https://docs.google.com/forms/d/e/1FAIpQLSfVumHC4jRMs9IMPbFYjr_mI8_k6ZHCwZCi4a_aER3rq2qCfA/viewform?usp=header) to obtain the link. After obtaining the link, download and extract the compressed file to [here](LLaMA-Factory/data/).
If you want to customize the dataset, you can refer to the [instructions](LLaMA-Factory/data/README.md).


## Download Pretrained Base SD Model
Select an SD model for incremental generation. Download to the "pretrained" directory.
- [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
- [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)

## Download Pretrained SD Weights
Download the weights of the pre-trained SD model that we have trained. Download to the "pretrained" directory.
- [ioky/SD3.5_large](https://huggingface.co/ioky/SD3.5_large)
- [ioky/SD3.5_medium](https://huggingface.co/ioky/SD3.5_medium)



## Download Pretrained Base VL Model
Select an VL model for multi-scale temporal prediction. Download to the "pretrained" directory.

- [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
- [OpenGVLab/InternVL3-8B-hf](https://huggingface.co/OpenGVLab/InternVL3-8B-hf)
- [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it)

## Download Pretrained LoRA Weights of VL Model
Download the LoRA weights of the pre-trained VL model that we have trained. Download to the "LoRA" directory.

- [ioky/Qwen2.5-VL-7B-Instruct](https://huggingface.co/ioky/Qwen2.5-VL-7B-Instruct)
- [ioky/InternVL3-8B-hf](https://huggingface.co/ioky/InternVL3-8B-hf)
- [ioky/gemma-3-4b-it](https://huggingface.co/ioky/gemma-3-4b-it)

## Temporal Prediction via Incremental Generation
```bash
cd MSTP/LLaMA-Factory
# use Qwen2.5-VL-7B-Instruct
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 --data_dir dir_to_dataset --sd_model large --mode test --model_name Qwen2.5-VL-7B-Instruct
# use gemma-3-4b-it
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 --data_dir dir_to_dataset --sd_model large --mode test --model_name gemma-3-4b-it
# use InternVL3-8B-hf
python ../TP_IG.py --cir 5 --time 1 --start 0 --end 200 --data_dir dir_to_dataset --sd_model large --mode test --model_name InternVL3-8B-hf

```
## SD Model Training
To train the SD3.5 model, please refer to [fine-tuning guide](https://stabilityai.notion.site/Stable-Diffusion-3-5-fine-tuning-guide-11a61cdcd1968027a15bdbd7c40be8c6)

## VL Model Training
This project uses LoRA for training.
```bash
cd /MSTP/LLaMA-Factory
DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/train_lora/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```
## VL Model Validation
Generate VL model results in batches
```bash
DISABLE_VERSION_CHECK=1 llamafactory-cli train examples/predict/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```
## VL Model Merge
```text
DISABLE_VERSION_CHECK=1 llamafactory-cli export examples/merge_lora/Qwen2.5-VL-7B-Instruct/qwen2.5vl_lora_sft_chain1_1s.yaml
```

## Questions
For further question about the code or paper, welcome to create an issue, or contact, please email yuanguojian@foxmail.com

## Citing MSTP
If you find this code useful for your research, please use the following BibTeX entries:
```bash
@misc{zeng2025multiscaletemporalpredictionincremental,
      title={Multi-scale Temporal Prediction via Incremental Generation and Multi-agent Collaboration}, 
      author={Zhitao Zeng and Guojian Yuan and Junyuan Mao and Yuxuan Wang and Xiaoshuang Jia and Yueming Jin},
      year={2025},
      eprint={2509.17429},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.17429}, 
}
```
