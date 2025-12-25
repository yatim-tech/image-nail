#!/usr/bin/env python3
"""
Unified image model training script using ai-toolkit for all model types (SDXL, Flux, Z-Image, Qwen-Image)
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import re
import time

import yaml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    """Get the actual model path, handling directories with single safetensors files"""
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path


def load_lrs_config(model_type: str, is_style: bool) -> dict:
    """Load the appropriate LRS configuration based on model type and training type"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def get_network_config_for_sdxl(model_name: str, is_style: bool) -> dict:
    """Get network configuration based on SDXL model and training type"""
    
    # Network configuration IDs for person training
    network_config_person = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 467,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 467,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 699,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 467,
        "dataautogpt3/TempestV0.1": 456,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 467,
        "fluently/Fluently-XL-Final": 228,
        "mann-e/Mann-E_Dreams": 456,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 228,
        "recoilme/colorfulxl": 228,
        "zenless-lab/sdxl-aam-xl-anime-mix": 456,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
        "zenless-lab/sdxl-anything-xl": 228,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
        "Corcelio/mobius": 228,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
    }

    # Network configuration for style training (all 235)
    network_config_style = {k: 235 for k in network_config_person.keys()}
    
    # Also add special case for TempestV0.1 in style
    network_config_style["dataautogpt3/TempestV0.1"] = 228

    # Config mapping to actual network parameters
    config_mapping = {
        228: {
            "network_dim": 32,
            "network_alpha": 32,
            "use_conv": False
        },
        235: {
            "network_dim": 32,
            "network_alpha": 32,
            "use_conv": True,
            "conv_dim": 4,
            "conv_alpha": 4
        },
        456: {
            "network_dim": 64,
            "network_alpha": 64,
            "use_conv": False
        },
        467: {
            "network_dim": 64,
            "network_alpha": 64,
            "use_conv": True,
            "conv_dim": 4,
            "conv_alpha": 4
        },
        699: {
            "network_dim": 96,
            "network_alpha": 96,
            "use_conv": True,
            "conv_dim": 4,
            "conv_alpha": 4
        },
    }

    # Select the appropriate config
    network_config_dict = network_config_style if is_style else network_config_person
    config_id = network_config_dict.get(model_name, 235)  # Default to 235
    
    return config_mapping[config_id]


def create_aitoolkit_config(task_id: str, model_path: str, model_name: str, model_type: str, 
                            expected_repo_name: str, trigger_word: str = None) -> str:
    """Create ai-toolkit configuration for any model type"""
    
    train_data_dir = train_paths.get_image_training_images_dir(task_id)
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Determine if this is style or person training
    _, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)
    
    # Base configuration structure
    config = {
        "job": "extension",
        "config": {
            "name": f"{model_type}_lora_{task_id}",
            "process": [
                {
                    "type": "sd_trainer" if model_type == "sdxl" else "diffusion_trainer",
                    "training_folder": output_dir,
                    "device": "cuda:0",
                    "network": {},
                    "save": {
                        "dtype": "bf16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 4,
                        "save_format": "safetensors"
                    },
                    "datasets": [
                        {
                            "folder_path": train_data_dir,
                            "caption_ext": "txt",
                            "cache_latents_to_disk": True,
                            "resolution": [512, 768, 1024]
                        }
                    ],
                    "train": {
                        "gradient_checkpointing": True,
                        "dtype": "bf16"
                    },
                    "model": {
                        "name_or_path": model_path
                    }
                }
            ]
        },
        "meta": {
            "name": f"{model_type}_lora",
            "version": "1.0"
        }
    }
    
    process = config["config"]["process"][0]
    
    # Add trigger word if provided
    if trigger_word:
        process["trigger_word"] = trigger_word
    
    # Model-specific configurations
    if model_type == "sdxl":
        # Get SDXL-specific network config
        network_config = get_network_config_for_sdxl(model_name, is_style)
        
        process["network"] = {
            "type": "lora",
            "linear": network_config["network_dim"],
            "linear_alpha": network_config["network_alpha"]
        }
        
        if network_config["use_conv"]:
            process["network"]["conv"] = network_config["conv_dim"]
            process["network"]["conv_alpha"] = network_config["conv_alpha"]
        
        # SDXL training parameters
        process["train"].update({
            "batch_size": 4,
            "steps": 1600,
            "gradient_accumulation_steps": 1,
            "train_unet": True,
            "train_text_encoder": True,
            "noise_scheduler": "ddpm",
            "optimizer": "adamw8bit",
            "lr": 0.0001,
            "min_snr_gamma": 5.0 if not is_style else 8.0,
        })
        
        # Load LRS config for learning rates
        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hashlib.sha256(model_name.encode('utf-8')).hexdigest()
            default_config = lrs_config.get("default", {})
            model_config = lrs_config.get("data", {}).get(model_hash, {})
            
            # Merge configs
            merged_lrs = {**default_config, **model_config}
            
            if "unet_lr" in merged_lrs:
                process["train"]["lr"] = merged_lrs["unet_lr"]
            if "text_encoder_lr" in merged_lrs:
                process["train"]["text_encoder_lr"] = merged_lrs["text_encoder_lr"]
            if "max_train_steps" in merged_lrs:
                process["train"]["steps"] = merged_lrs["max_train_steps"]
        
        process["model"]["is_xl"] = True
        
    elif model_type == "flux":
        # Flux configuration
        process["network"] = {
            "type": "lora",
            "linear": 128,
            "linear_alpha": 128
        }
        
        process["train"].update({
            "batch_size": 1,
            "steps": 2000,
            "gradient_accumulation_steps": 1,
            "train_unet": True,
            "train_text_encoder": False,
            "noise_scheduler": "flowmatch",
            "optimizer": "adamw8bit",
            "lr": 0.0001,
        })
        
        # Load LRS config for Flux
        lrs_config = load_lrs_config(model_type, is_style)
        if lrs_config:
            model_hash = hashlib.sha256(model_name.encode('utf-8')).hexdigest()
            default_config = lrs_config.get("default", {})
            model_config = lrs_config.get("data", {}).get(model_hash, {})
            
            merged_lrs = {**default_config, **model_config}
            
            if "unet_lr" in merged_lrs:
                process["train"]["lr"] = merged_lrs["unet_lr"]
            if "text_encoder_lr" in merged_lrs:
                process["train"]["text_encoder_lr"] = merged_lrs["text_encoder_lr"]
            if "max_train_steps" in merged_lrs:
                process["train"]["steps"] = merged_lrs["max_train_steps"]
        
        process["model"]["is_flux"] = True
        process["model"]["quantize"] = True
        
    elif model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]:
        # Z-Image / Qwen-Image configuration
        process["network"] = {
            "type": "lora",
            "linear": 32,
            "linear_alpha": 32,
            "conv": 16,
            "conv_alpha": 16
        }
        
        process["train"].update({
            "batch_size": 4,
            "steps": 2000,
            "lr": 0.0001,
            "optimizer": "adamw8bit",
            "noise_scheduler": "flowmatch",
            "timestep_type": "weighted"
        })
        
        # Model-specific settings
        if model_type == ImageModelType.Z_IMAGE.value:
            process["model"]["arch"] = "zimage:turbo"
            process["model"]["quantize"] = True
            process["model"]["qtype"] = "qfloat8"
            process["model"]["quantize_te"] = True
            process["model"]["qtype_te"] = "qfloat8"
            process["model"]["assistant_lora_path"] = "/cache/hf_cache/zimage_turbo_training_adapter_v2.safetensors"
        elif model_type == ImageModelType.QWEN_IMAGE.value:
            process["model"]["arch"] = "qwen2vl"
            process["model"]["quantize"] = True
            process["model"]["qtype"] = "qfloat8"
    
    # Save configuration
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
    save_config(config, config_path)
    
    print(f"Created ai-toolkit config at {config_path}", flush=True)
    print(f"Config content:\n{yaml.dump(config, default_flow_style=False)}", flush=True)
    
    return config_path


def run_training(model_type: str, config_path: str):
    """Run training using ai-toolkit for all model types"""
    print(f"Starting ai-toolkit training with config: {config_path}", flush=True)

    training_command = [
        "python3",
        "/app/ai-toolkit/run.py",
        config_path
    ]

    try:
        print("Starting ai-toolkit training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")


async def main():
    print("---STARTING UNIFIED AI-TOOLKIT IMAGE TRAINING SCRIPT---", flush=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Unified Image Model Training Script (ai-toolkit)")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, 
                       choices=["sdxl", "flux", "z-image", "qwen-image"], 
                       help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--trigger-word", help="Trigger word for the training")
    parser.add_argument("--hours-to-complete", type=float, required=True, 
                       help="Number of hours to complete the task")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)
    
    model_path = train_paths.get_image_base_model_path(args.model)

    # Prepare dataset
    print("Preparing dataset...", flush=True)
    training_images_repeat = (cst.DIFFUSION_SDXL_REPEATS 
                              if args.model_type == ImageModelType.SDXL.value 
                              else cst.DIFFUSION_FLUX_REPEATS)
    
    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=training_images_repeat,
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # Create ai-toolkit config file
    config_path = create_aitoolkit_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        args.trigger_word
    )

    # Run training with ai-toolkit
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())

