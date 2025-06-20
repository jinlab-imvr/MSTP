#!/usr/bin/env python3
import os
import gc
import math
import random
import json
import argparse

import torch
import torch._dynamo
from torch.multiprocessing import Pool, set_start_method
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

from tqdm import tqdm
from difflib import get_close_matches

from transformers import (
    AutoModelForImageTextToText,
    Qwen2_5_VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    AutoProcessor
)
from diffusers import StableDiffusion3Img2ImgPipeline, BitsAndBytesConfig
from safetensors.torch import load_file
from peft import PeftModel, PeftConfig

from qwen_vl_utils import process_vision_info

torch._dynamo.config.cache_size_limit = 64  # Optional: increase limit
torch._dynamo.disable()
torch.set_float32_matmul_precision('high')

class VLProcessor:
    """
    A unified processor class for handling image and text prompts using Qwen2.5-VL and related vision-language models.

    This class enables loading and inference with various vision-language models (including Qwen2.5-VL, InternVL3, Gemma3, etc.),
    with optional support for LoRA fine-tuned weights. It provides a consistent interface for generating textual outputs
    from an image path and a text prompt.

    The class manages model and processor initialization, device placement, and memory/resource cleanup.
    It is designed for efficient repeated or large-scale inference, with methods to release GPU and system memory after use.

    Args:
        model_name (str): Name of the vision-language model (e.g., "Qwen2.5-VL-7B-Instruct").
        lora_path (str): Path to the LoRA fine-tuned weights.
        base_model_path (str): Path to the base model.
        device (str): Device for model execution ("auto", "cuda", "cpu", etc.).
        use_lora (bool): Whether to use LoRA fine-tuned weights. If True, lora_path should be provided.
    """
    
    def __init__(self, model_name, lora_path, base_model_path, device="auto", use_lora=True):
        try:
            if not os.path.exists(base_model_path):
                raise FileNotFoundError(f"Model path not found: {base_model_path}")
            self.model_name = model_name
            if use_lora is not None:
                if not os.path.exists(lora_path):
                    raise FileNotFoundError(f"Model path not found: {lora_path}")
                if model_name == "InternVL3-8B-hf" or model_name == "InternVL3-38B-hf":
                    from transformers import AutoModelForImageTextToText, AutoProcessor
                    base_model = AutoModelForImageTextToText.from_pretrained(
                        base_model_path, device_map=device, torch_dtype=torch.bfloat16
                    )
                    self.model = PeftModel.from_pretrained(base_model, lora_path)
                    self.processor = AutoProcessor.from_pretrained(base_model_path)
                elif model_name == "Qwen2.5-VL-7B-Instruct" or model_name == "Qwen2.5-VL-32B-Instruct":
                    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        base_model_path, device_map=device, torch_dtype=torch.bfloat16
                    )
                    self.model = PeftModel.from_pretrained(base_model, lora_path)
                    self.processor = AutoProcessor.from_pretrained(base_model_path)
                elif model_name == "gemma-3-4b-it" or model_name == "gemma-3-27b-it":
                    from transformers import Gemma3ForConditionalGeneration, AutoProcessor
                    base_model = Gemma3ForConditionalGeneration.from_pretrained(
                        base_model_path, device_map=device, torch_dtype=torch.bfloat16
                    )
                    self.model = PeftModel.from_pretrained(base_model, lora_path)
                    self.processor = AutoProcessor.from_pretrained(base_model_path)
                else:
                    raise ValueError(f"Unsupported model name: {model_name}")
            else:
                if model_name == "InternVL3-8B-hf" or model_name == "InternVL3-38B-hf":
                    self.model = AutoModelForImageTextToText.from_pretrained(base_model_path, device_map=device, torch_dtype=torch.bfloat16)
                    self.processor = AutoProcessor.from_pretrained(base_model_path)
                elif model_name == "Qwen2.5-VL-7B-Instruct" or model_name == "Qwen2.5-VL-32B-Instruct":
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_path, device_map=device, torch_dtype=torch.bfloat16)
                    self.processor = AutoProcessor.from_pretrained(base_model_path)
                elif model_name == "gemma-3-4b-it" or model_name == "gemma-3-27b-it":
                    self.model = Gemma3ForConditionalGeneration.from_pretrained(base_model_path, device_map=device, torch_dtype=torch.bfloat16).eval()
                    self.processor = AutoProcessor.from_pretrained(base_model_path)
        except Exception as e:
            raise RuntimeError(f"Error initializing Qwen model: {str(e)}")

    def generate(self, image_path, text_prompt, max_new_tokens=128):
        """
        Generate text output using the Qwen2.5-VL model for a given image and text prompt.
        
        Args:
            image_path (str): Path to the input image
            text_prompt (str): Text prompt to accompany the image
            max_new_tokens (int): Maximum number of tokens to generate (default: 128)
        
        Returns:
            str: Generated text output
        
        Raises:
            FileNotFoundError: If image_path is invalid
            RuntimeError: If generation fails
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image path not found: {image_path}")
            if self.model_name == "InternVL3-8B-hf" or self.model_name == "InternVL3-38B-hf":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": image_path},
                            {"type": "text", "text": text_prompt},
                        ],
                    }
                ]
                inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device, dtype=torch.bfloat16)

                generate_ids = self.model.generate(**inputs, max_new_tokens=50)
                decoded_output = self.processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            elif self.model_name == "Qwen2.5-VL-7B-Instruct" or self.model_name == "Qwen2.5-VL-32B-Instruct":
                messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {"type": "text", "text": text_prompt},
                            ],
                        }
                    ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                decoded_output = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            elif self.model_name == "gemma-3-4b-it" or self.model_name == "gemma-3-27b-it":
                messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": text_prompt}
                        ]
                    }
                ]

                inputs = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt"
                ).to(self.model.device, dtype=torch.bfloat16)

                input_len = inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
                    generation = generation[0][input_len:]

                decoded_output = self.processor.decode(generation, skip_special_tokens=True)
                
            return decoded_output


        except Exception as e:
            raise RuntimeError(f"Error generating with Qwen model: {str(e)}")
    
    def release(self):
        """
        Release the model and processor, freeing up memory.
        
        This method deletes the model and processor, clears the PyTorch cache,
        and forces garbage collection to release GPU and system memory.
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
        
        except Exception as e:
            raise RuntimeError(f"Error releasing Qwen model resources: {str(e)}")

class ImageGenerator:
    def __init__(self, model_path="pretrained/stable-diffusion-3.5-large",
                 weights_path="SD_grasp/SD3.5_large/diffusion_pytorch_model.safetensors",
                 device="cuda:0"):
        """
        Initialize the ImageGenerator with model and weights paths.
        
        Args:
            model_path (str): Path or identifier for the pretrained model
            weights_path (str): Path to the safetensors weights file
            device (str): Device to run the model on (default: "cuda")
        """
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        
        self._load_weights(weights_path)
        
        # Enable CPU offloading
        # self.pipe.enable_sequential_cpu_offload()
        
        self.num_inference_steps = 40
        self.guidance_scale = 9.2

    def _load_weights(self, weights_path):
        """Load weights from safetensors file into the transformer."""
        state_dict = load_file(weights_path)
        self.pipe.transformer.load_state_dict(state_dict)

    def generate_image(self, prompt, init_image, 
                      output_path='output/image.png'):
        """
        Generate an image based on prompt and initial image.
        
        Args:
            prompt (str): Text prompt for image generation
            init_image_path (str): Path to the initial image
            output_path (str): Path to save the generated image
        
        Returns:
            tuple: (initial_image, generated_image)
        """
        generated_image = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            image=init_image,
        ).images[0]
        try:
            generated_image.save(output_path, format="JPEG")
        except Exception as e:
            pass
        print(f"Saved generated image to {output_path}")
        return generated_image

    def release(self):
        """
        Release the pipeline and free GPU memory.
        
        This method deletes the pipeline, clears the PyTorch cache,
        and forces garbage collection to release GPU and system memory.
        """
        try:
            if hasattr(self, 'pipe') and self.pipe is not None:
                del self.pipe
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            gc.collect()
        except Exception as e:
            raise RuntimeError(f"Error releasing ImageGenerator resources: {str(e)}")
class PSDataProcessor:
    def __init__(self, mode="test"):
        self.mode = mode
        self.train_data_path = "data/unified_labels_image_train_phase_step.json"
        self.test_data_path = "data/unified_labels_image_test_phase_step.json"
        self.total_cases = {
            "train": ['CASE001', 'CASE002', 'CASE003', 'CASE004', 'CASE007', 'CASE014', 'CASE015', 'CASE021'],
            "test": ['CASE041', 'CASE047', 'CASE050', 'CASE051', 'CASE053']
        }
        self.data_path = self.train_data_path if self.mode == "train" else self.test_data_path
        self.totle_ps_dict = self._process_data()
    
    def _process_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        case_dicts = {}
        for i in data:
            case = i["image"].split('/')[3]
            case_dicts.setdefault(case, []).append(i)
        
        totle_ps_list = []
        for c in case_dicts.keys():
            ps_dict = {}
            for i in case_dicts[c]:
                text = i["conversations"][1]["value"]
                phase = text.split("**")[1]
                step = text.split("**")[3]
                key = f"{phase}, {step}".lower()
                ps_dict.setdefault(key, []).append(text)
            totle_ps_list.append(ps_dict)
        
        totle_ps_dict = {self.total_cases[self.mode][i]: totle_ps_list[i] for i in range(len(totle_ps_list))}
        return totle_ps_dict
    
    def get_full_text(self, case, ps):
        return random.choice(self.totle_ps_dict.get(case, {}).get(ps, ["No data available"]))

class Refiner:
    def __init__(self, totle_ps_dict):
        self.totle_ps_dict = totle_ps_dict
    def process(self, ps, case):
        try:
            phase = ps.split(", ")[0]
            step = ps.split(", ")[1]
            ps = phase + ', ' + step
        except IndexError:
            return None
        keys = self.totle_ps_dict[case].keys()
        if ps in keys:
            return ps
        else:
            for key in keys:
                if ps in key:
                    return key
            closest_key = get_close_matches(ps, keys, n=1)
            if closest_key:
                return closest_key[0]
            return None

# Function to load data and set paths based on mode
def load_data_and_paths(mode):
    train_data_path = "data/unified_labels_image_train_phase_step.json"
    test_data_path = "data/unified_labels_image_test_phase_step.json"
    
    if mode == "train":
        data_path = train_data_path
        totle_case = ['CASE001', 'CASE002', 'CASE003', 'CASE004', 'CASE007', 'CASE014', 'CASE015', 'CASE021']
    else:
        data_path = test_data_path
        totle_case = ['CASE041', 'CASE047', 'CASE050', 'CASE051', 'CASE053']
    
    with open(data_path, "r") as f:
        data = json.load(f)
    
    case_dicts = {}
    for i in data:
        case = i["image"].split('/')[3]
        case_dicts.setdefault(case, []).append(i)
    
    totle_ps_list = []
    for c in case_dicts.keys():
        ps_dict = {}
        for i in case_dicts[c]:
            case = i["image"].split('/')[3]
            text = i["conversations"][1]["value"]
            phase = text.split("**")[1]
            step = text.split("**")[3]
            ps_dict.setdefault(phase + ', ' + step, []).append(text)
        totle_ps_list.append(ps_dict)
    
    totle_ps_dict = {totle_case[i]: totle_ps_list[i] for i in range(len(totle_ps_list))}
    return totle_ps_dict, totle_case

# Function to split phases and steps
def spilt_ps(totle_ps_dict, case):
    totle_p_list = [i.split(',')[0] for i in totle_ps_dict[case].keys()]
    totle_p_list = list(set(totle_p_list))
    totle_p_dict = {i: [] for i in totle_p_list}
    for i in totle_ps_dict[case].keys():
        p = i.split(', ')[0]
        s = i.replace(p + ', ', '')
        totle_p_dict[p].append(s)
    return totle_p_dict

# Function to format shuffled multiple-choice options
def format_shuffled_mcq(steps):
    random.shuffle(steps)
    formatted_text = ". ".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
    return formatted_text
    
# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Run image generation with configurable parameters.")
    parser.add_argument("--cir", type=int, default=5, help="Number of iterations")
    parser.add_argument("--time", type=int, default=1, help="Time parameter for chain paths")
    parser.add_argument("--start", type=int, default=0, help="Start index for data slicing")
    parser.add_argument("--end", type=int, default=500, help="Number of data samples to process")
    parser.add_argument("--sd_model", type=str, default="large", choices=["large", "medium"], help="Stable Diffusion model type")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"], help="Mode: train or test dataset")
    parser.add_argument("--model_name", type=str, default="Qwen2.5-VL-7B-Instruct", choices=["gemma-3-4b-it", "gemma-3-27b-it", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "InternVL3-8B-hf", "InternVL3-38B-hf"],
                        help="Model name to use for processing")
    parser.add_argument("--save_dir", type=str, default="DM/test", help="Directory to save results")
    parser.add_argument("--restrain", action="store_true", help="Use restrain prompt (default: False). Set to True to use restrain prompt.")
    return parser.parse_args()

# Main execution
if __name__ == "__main__":
    args = parse_args()
    CIR = args.cir
    TIME = args.time
    START = args.start
    END = args.end
    SD_MODEL = args.sd_model
    MODE = args.mode
    MODEL_NAME = args.model_name
    RESTRAIN = args.restrain
    save_dir = os.path.join(args.save_dir, f"{MODEL_NAME}_{TIME}s_{SD_MODEL}")

    # Load data and paths
    data_path = f"data/unified_labels_image_test_phase_step_chain_1_{TIME}s.json"
    with open(data_path, "r") as f:
        data = json.load(f)
    
    totle_ps_dict, totle_case = load_data_and_paths(MODE)
    totle_result = []
    data = data[START:END]
    
    PSprocessor = PSDataProcessor(mode=MODE)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if SD_MODEL == "large":
        model_path = "pretrained/stable-diffusion-3.5-large" # "stabilityai/stable-diffusion-3.5-large" 
        weights_path = "pretrained/SD3.5_large/diffusion_pytorch_model.safetensors" # ioky/SD3.5_large
    elif SD_MODEL == "medium":
        model_path = "pretrained/stable-diffusion-3.5-medium" # "stabilityai/stable-diffusion-3.5-medium"
        weights_path = "pretrained/SD3.5_medium/diffusion_pytorch_model.safetensors" # ioky/SD3.5_medium
    generator = ImageGenerator(model_path=model_path, weights_path=weights_path, device="balanced")
    chain1_path = f"saves/{MODEL_NAME}/{MODEL_NAME}_chain1_{TIME}s/lora/sft"
    base_model_path = f"pretrained/{MODEL_NAME}"
    vl_processor = VLProcessor(
        model_name=MODEL_NAME,
        lora_path=chain1_path,           # LoRA weight path
        base_model_path=base_model_path,     
        device="auto",
        use_lora=True
    )
    chain2_path = f"saves/{MODEL_NAME}/{MODEL_NAME}_chain2_{TIME}s/lora/sft"
    vl_processor2 = VLProcessor(model_name=MODEL_NAME, lora_path=chain2_path, base_model_path=base_model_path, device="auto", use_lora=True)
    chain3_path = f"saves/{MODEL_NAME}/{MODEL_NAME}_chain3_{TIME}s/lora/sft"
    vl_processor3 = VLProcessor(model_name=MODEL_NAME, lora_path=chain3_path, base_model_path=base_model_path, device="auto", use_lora=True)
    for i in range(len(data)):
        ori_prompt = data[i]["conversations"][0]["value"].replace("<image>", "")
        if RESTRAIN:
            temp_text_prompt = ori_prompt + " . Just answer 'Change' or 'Maintain' without analysis."
        else:
            temp_text_prompt = ori_prompt
        temp_image_path = data[i]["images"][0]
        case = data[i]["image"].split('/')[3] if "image" in data[i] else totle_case[0]  # Fallback to first case if not found
        
        for j in range(CIR):
            if os.path.exists(os.path.join(save_dir, f"{i + START}_{j + CIR}.txt")):
                break
            # Chain 1
            result = vl_processor.generate(temp_image_path, temp_text_prompt)
            result = result.replace(".", "")
            print("Generated Output 1:", result)
            if result == "Maintain":
                with open(os.path.join(save_dir, f"{i + START}_{j + 1}.txt"), 'w', encoding='utf-8') as file:
                    file.write(ori_prompt)
                print("The current phase: Maintain and gt:")
                continue
            
            current_ps = data[i]["current_ps"]
            phase = current_ps.split(", ")[0]
            spilt_ps_dict = spilt_ps(totle_ps_dict, case)
            totle_p_dict = list(set(spilt_ps_dict.keys()))
            random_keys = format_shuffled_mcq(totle_p_dict)
            
            # Chain 2
            if RESTRAIN:
                text_prompt2 = f"The current phase: {phase}. Choices: {random_keys}. Just answer without analysis, No need for numerical sequence."
            else:
                text_prompt2 = f"The current phase: {phase}. Choices: {random_keys}"

            pre_phase = vl_processor2.generate(temp_image_path, text_prompt2)
            pre_phase = pre_phase.replace(".", "")
            print("Generated Output2:", pre_phase)
            
            match = get_close_matches(pre_phase, totle_p_dict, n=1, cutoff=0.4)
            if match != []:
                pre_phase = match[0]
            # Chain 3
            random_keys = format_shuffled_mcq(spilt_ps_dict.get(pre_phase, []))
            if RESTRAIN:
                text = f"{pre_phase}. Choices: {random_keys}. Just answer without analysis, No need for numerical sequence."
            else:
                text = f"{pre_phase}. Choices: {random_keys}"
            pre_step = vl_processor3.generate(temp_image_path, text)
            pre_step = pre_step.replace(".", "")
            match = get_close_matches(pre_step, spilt_ps_dict.get(pre_phase, []), n=1, cutoff=0.4)
            if match!= []:
                pre_step = match[0]
            print("Generated Output3:", pre_step)
            pre_ps = pre_phase + ", " + pre_step
            totle_result.append(pre_ps)
            print("save the result:", pre_ps)
            with open(os.path.join(save_dir, f"{i + START}_{j + 1}.txt"), 'w', encoding='utf-8') as file:
                file.write(pre_ps)
            if j == CIR - 1:
                continue
            # Image Generation
            temp_prompt = PSprocessor.get_full_text(case, pre_ps.lower())
            print(temp_prompt)
            init_image = Image.open(temp_image_path).convert("RGB")
            save_path = os.path.join(save_dir, f'{i + START}_{j + 1}.jpg')
            generated_image = generator.generate_image(
                prompt=temp_prompt,
                init_image=init_image,
                output_path=save_path
            )
            print("Saved to", save_path)
            
            temp_image_path = save_path
            if RESTRAIN: 
                temp_text_prompt = pre_ps  + " . Just answer 'Change' or 'Maintain' without analysis."
            else:
                temp_text_prompt = pre_ps