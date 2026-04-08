#!/usr/bin/env python
# coding=utf-8
# Concept-token training script for SD3.5 Medium.
# Based on train_stage2_v2_sd3_baseline.py.
# Key additions vs. baseline:
#   1. Register a new learnable token (<new1> by default) into CLIP-L and CLIP-G tokenizers
#      and initialize its embedding from the dataset's `instance_word`.
#   2. Only the new token's embedding rows are optimised (gradient masking after backward).
#   3. A separate --embedding_lr controls the embedding learning rate.
#   4. Learned embeddings are saved as `learned_embeds.pt` at every checkpoint.
#   5. [v] in prompts is replaced with the registered new concept token at runtime.
#   NOTE: T5 is intentionally excluded – it uses SentencePiece and does not support
#         add_tokens(); the SD3.5 joint attention also makes per-layer embedding indexing
#         (EDLoRA style) unnecessary.

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from peft import get_peft_model

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from PureCC_dataset import PureCC


if is_wandb_available():
    import wandb

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Concept-token helpers
# ---------------------------------------------------------------------------

def init_concept_tokens(
    tokenizer_one,
    tokenizer_two,
    text_encoder_one,
    text_encoder_two,
    new_concept_token,
    initializer_token,
):
    """
    Register `new_concept_token` in both CLIP tokenizers and initialise its
    embedding from `initializer_token` (an existing vocabulary word).

    Returns:
        new_token_id_one (int): token index in tokenizer_one / text_encoder_one
        new_token_id_two (int): token index in tokenizer_two / text_encoder_two
    """
    # --- CLIP-L (tokenizer_one / text_encoder_one) ---
    num_added_one = tokenizer_one.add_tokens([new_concept_token])
    assert num_added_one == 1, f"Token '{new_concept_token}' was already present in tokenizer_one"
    new_token_id_one = tokenizer_one.convert_tokens_to_ids(new_concept_token)
    text_encoder_one.resize_token_embeddings(len(tokenizer_one))

    # --- CLIP-G (tokenizer_two / text_encoder_two) ---
    num_added_two = tokenizer_two.add_tokens([new_concept_token])
    assert num_added_two == 1, f"Token '{new_concept_token}' was already present in tokenizer_two"
    new_token_id_two = tokenizer_two.convert_tokens_to_ids(new_concept_token)
    text_encoder_two.resize_token_embeddings(len(tokenizer_two))

    # --- Initialise from existing word ---
    with torch.no_grad():
        # CLIP-L
        init_ids_one = tokenizer_one.encode(initializer_token, add_special_tokens=False)
        if len(init_ids_one) == 0:
            logger.warning(
                f"initializer_token '{initializer_token}' is unknown in tokenizer_one; "
                "using random init instead."
            )
            init_embed_one = torch.randn_like(
                text_encoder_one.get_input_embeddings().weight[0]
            ) * 0.017
        else:
            init_embed_one = text_encoder_one.get_input_embeddings().weight[init_ids_one[0]].clone()
        text_encoder_one.get_input_embeddings().weight[new_token_id_one] = init_embed_one

        # CLIP-G
        init_ids_two = tokenizer_two.encode(initializer_token, add_special_tokens=False)
        if len(init_ids_two) == 0:
            logger.warning(
                f"initializer_token '{initializer_token}' is unknown in tokenizer_two; "
                "using random init instead."
            )
            init_embed_two = torch.randn_like(
                text_encoder_two.get_input_embeddings().weight[0]
            ) * 0.017
        else:
            init_embed_two = text_encoder_two.get_input_embeddings().weight[init_ids_two[0]].clone()
        text_encoder_two.get_input_embeddings().weight[new_token_id_two] = init_embed_two

    logger.info(
        f"Registered concept token '{new_concept_token}': "
        f"id_one={new_token_id_one}, id_two={new_token_id_two}, "
        f"initializer='{initializer_token}'"
    )
    return new_token_id_one, new_token_id_two


def zero_out_non_concept_grads(embedding_layer, concept_token_id):
    """Zero gradient rows in embedding_layer.weight for every token except concept_token_id."""
    if embedding_layer.weight.grad is None:
        return
    grad = embedding_layer.weight.grad
    n_tokens = grad.shape[0]
    mask = torch.ones(n_tokens, dtype=torch.bool, device=grad.device)
    mask[concept_token_id] = False
    grad[mask] = 0.0


# ---------------------------------------------------------------------------
# Helpers copied / kept from baseline
# ---------------------------------------------------------------------------

def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3",
        revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def log_validation(pipeline, args, accelerator, pipeline_args, epoch, torch_dtype, is_final_validation=False):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    autocast_ctx = nullcontext()
    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]
    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {phase_name: [wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)]}
            )
    del pipeline
    free_memory()
    return images


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def _encode_prompt_with_t5(text_encoder, tokenizer, max_sequence_length, prompt=None,
                            num_images_per_prompt=1, device=None, text_input_ids=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, add_special_tokens=True, return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds


def _encode_prompt_with_clip(text_encoder, tokenizer, prompt: str, device=None,
                              text_input_ids=None, num_images_per_prompt: int = 1):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)
    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return prompt_embeds, pooled_prompt_embeds


def encode_prompt(text_encoders, tokenizers, prompt: str, max_sequence_length,
                  device=None, num_images_per_prompt: int = 1, text_input_ids_list=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]
    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[i] if text_input_ids_list else None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)
    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)
    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1], tokenizers[-1], max_sequence_length,
        prompt=prompt, num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[-1] if text_input_ids_list else None,
        device=device if device is not None else text_encoders[-1].device,
    )
    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
    return prompt_embeds, pooled_prompt_embeds


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="SD3.5 concept-token + LoRA training.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/cfs/cfs-1dafgugv/connorxian/hf_cache/SD3.5-medium",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument(
        "--new_concept_token",
        type=str,
        default="<new1>",
        help="The special token string to register as the new concept (e.g. '<new1>').",
    )
    parser.add_argument(
        "--embedding_lr",
        type=float,
        default=5e-3,
        help="Learning rate for the new concept token embedding (separate from LoRA lr).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/cfs/cfs-kuxuxpyv/wsliu/connorxian/SPF_dataset/v2",
    )
    parser.add_argument("--csv_name", type=str, default="catroon1_style.csv")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--class_data_dir", type=str, default=None)
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="A [v]",
        help="Template prompt; [v] is replaced with --new_concept_token at runtime.",
    )
    parser.add_argument("--class_prompt", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=77)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--num_validation_images", type=int, default=4)
    parser.add_argument("--validation_epochs", type=int, default=50)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--with_prior_preservation", default=False, action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0)
    parser.add_argument("--num_class_images", type=int, default=100)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/cfs/cfs-1dafgugv/connorxian/SPF_output/v2/concept_token_sd3",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--center_crop", default=False, action="store_true")
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--sample_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=400)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the transformer LoRA parameters.")
    parser.add_argument("--text_encoder_lr", type=float, default=5e-6,
                        help="(Unused in this script; use --embedding_lr instead.)")
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--precondition_outputs", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--prodigy_beta3", type=float, default=None)
    parser.add_argument("--prodigy_decouple", type=bool, default=True)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04)
    parser.add_argument("--adam_weight_decay_text_encoder", type=float, default=1e-03)
    parser.add_argument("--lora_layers", type=str, default=None)
    parser.add_argument("--lora_blocks", type=str, default=None)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True)
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--cache_latents", action="store_true", default=False)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--upcast_before_saving", action="store_true", default=False)
    parser.add_argument("--prior_generation_precision", type=str, default=None,
                        choices=["no", "fp32", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError("Cannot use both --report_to=wandb and --hub_token.")

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("bf16 mixed precision is not supported on MPS.")

    args.output_dir = os.path.join(args.output_dir, args.csv_name.split(".")[0])
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # -----------------------------------------------------------------------
    # Load tokenizers and text encoder classes
    # -----------------------------------------------------------------------
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_3", revision=args.revision,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        revision=args.revision, variant=args.variant,
    )

    # Freeze everything by default
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)

    # -----------------------------------------------------------------------
    # Mixed precision dtype
    # -----------------------------------------------------------------------
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("bf16 not supported on MPS.")

    vae.to(accelerator.device, dtype=torch.float32)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # -----------------------------------------------------------------------
    # Build dataset early so we can read the initializer_token from instance_word
    # -----------------------------------------------------------------------
    train_dataset = PureCC(
        data_path=args.data_path,
        csv_name=args.csv_name,
        size=args.resolution,
        repeats=args.repeats,
        custom_instance_prompts=True,
    )

    # Use the first sample's instance_word as the concept initializer
    first_sample = train_dataset[0]
    initializer_token = first_sample["instance_word"]
    logger.info(f"Using instance_word='{initializer_token}' as concept token initializer.")

    # -----------------------------------------------------------------------
    # Register concept token and initialise embeddings
    # -----------------------------------------------------------------------
    new_token_id_one, new_token_id_two = init_concept_tokens(
        tokenizer_one, tokenizer_two,
        text_encoder_one, text_encoder_two,
        args.new_concept_token, initializer_token,
    )

    # Enable gradient only for the embedding layers of CLIP-L and CLIP-G.
    # Keep them in float32 regardless of mixed precision for stable optimisation.
    text_encoder_one.get_input_embeddings().requires_grad_(True)
    text_encoder_two.get_input_embeddings().requires_grad_(True)
    text_encoder_one.get_input_embeddings().to(torch.float32)
    text_encoder_two.get_input_embeddings().to(torch.float32)

    # -----------------------------------------------------------------------
    # LoRA config for transformer
    # -----------------------------------------------------------------------
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
    if args.lora_blocks is not None:
        target_blocks = [int(block.strip()) for block in args.lora_blocks.split(",")]
        target_modules = [
            f"transformer_blocks.{block}.{module}"
            for block in target_blocks for module in target_modules
        ]

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer = get_peft_model(transformer, transformer_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # -----------------------------------------------------------------------
    # Save / load hooks
    # -----------------------------------------------------------------------
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            learned_embeds_one = None
            learned_embeds_two = None

            for model in models:
                unwrapped = unwrap_model(model)
                if isinstance(unwrapped, type(unwrap_model(transformer))):
                    # Save transformer LoRA
                    m = unwrapped
                    if args.upcast_before_saving:
                        m = m.to(torch.float32)
                    transformer_lora_layers_to_save = get_peft_model_state_dict(m)
                    m.save_pretrained(output_dir)
                elif isinstance(unwrapped, type(unwrap_model(text_encoder_one))):
                    # Distinguish by hidden_size
                    hidden_size = unwrapped.config.hidden_size
                    if hidden_size == 768:
                        learned_embeds_one = (
                            unwrapped.get_input_embeddings()
                            .weight[new_token_id_one]
                            .detach()
                            .cpu()
                            .float()
                        )
                    elif hidden_size == 1280:
                        learned_embeds_two = (
                            unwrapped.get_input_embeddings()
                            .weight[new_token_id_two]
                            .detach()
                            .cpu()
                            .float()
                        )
                    else:
                        logger.warning(f"Unexpected hidden_size={hidden_size} in text encoder; skipping.")
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                if weights:
                    weights.pop()

            # Save learned embeddings
            if learned_embeds_one is not None or learned_embeds_two is not None:
                torch.save(
                    {
                        "new_concept_token": args.new_concept_token,
                        "initializer_token": initializer_token,
                        "embedding_one": learned_embeds_one,   # CLIP-L
                        "embedding_two": learned_embeds_two,   # CLIP-G
                    },
                    os.path.join(output_dir, "learned_embeds.pt"),
                )
                logger.info(f"Saved learned_embeds.pt to {output_dir}")

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()
                if isinstance(unwrap_model(model), type(unwrap_model(transformer))):
                    transformer_ = unwrap_model(model)
                elif isinstance(unwrap_model(model), type(unwrap_model(text_encoder_one))):
                    hidden_size = unwrap_model(model).config.hidden_size
                    if hidden_size == 768:
                        text_encoder_one_ = unwrap_model(model)
                    else:
                        text_encoder_two_ = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected model: {model.__class__}")
        else:
            transformer_ = SD3Transformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="transformer"
            )
            transformer_.add_adapter(transformer_lora_config)

        lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

        # Restore learned embeddings
        embeds_path = os.path.join(input_dir, "learned_embeds.pt")
        if os.path.exists(embeds_path):
            saved = torch.load(embeds_path, map_location="cpu")
            if text_encoder_one_ is not None and saved.get("embedding_one") is not None:
                text_encoder_one_.get_input_embeddings().weight.data[new_token_id_one] = (
                    saved["embedding_one"].to(text_encoder_one_.device)
                )
            if text_encoder_two_ is not None and saved.get("embedding_two") is not None:
                text_encoder_two_.get_input_embeddings().weight.data[new_token_id_two] = (
                    saved["embedding_two"].to(text_encoder_two_.device)
                )
            logger.info(f"Restored learned embeddings from {embeds_path}")

        if args.mixed_precision == "fp16":
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps
            * args.train_batch_size * accelerator.num_processes
        )

    # Upcast LoRA params to fp32 for fp16 training
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # -----------------------------------------------------------------------
    # Optimizer parameter groups
    # -----------------------------------------------------------------------
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Embedding parameters (float32, two separate entries so we can control lr)
    embedding_parameters = (
        list(text_encoder_one.get_input_embeddings().parameters())
        + list(text_encoder_two.get_input_embeddings().parameters())
    )

    params_to_optimize = [
        {"params": transformer_lora_parameters, "lr": args.learning_rate},
        {
            "params": embedding_parameters,
            "lr": args.embedding_lr,
            "weight_decay": 0.0,   # no weight decay for embeddings
        },
    ]

    # -----------------------------------------------------------------------
    # Optimizer creation
    # -----------------------------------------------------------------------
    if not (args.optimizer.lower() in ("prodigy", "adamw")):
        logger.warning(f"Unsupported optimizer '{args.optimizer}', falling back to adamw.")
        args.optimizer = "adamw"

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("Install bitsandbytes to use 8-bit Adam.")
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("Install prodigyopt to use Prodigy.")
        optimizer_class = prodigyopt.Prodigy
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # -----------------------------------------------------------------------
    # Dataloader
    # -----------------------------------------------------------------------
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # T5 is always frozen and only used for inference – no gradient needed.
    tokenizers_all = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders_all = [text_encoder_one, text_encoder_two, text_encoder_three]

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor

    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
        if args.validation_prompt is None:
            del vae
            free_memory()

    # -----------------------------------------------------------------------
    # Scheduler
    # -----------------------------------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # -----------------------------------------------------------------------
    # Accelerator prepare
    # We prepare transformer + both CLIP text encoders (since their embeddings
    # are trainable) + optimizer + dataloader + lr_scheduler.
    # T5 is kept outside accelerator (frozen inference only).
    # -----------------------------------------------------------------------
    (
        transformer,
        text_encoder_one,
        text_encoder_two,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        transformer, text_encoder_one, text_encoder_two,
        optimizer, train_dataloader, lr_scheduler,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-sd3-concept-token", config=vars(args))

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running concept-token training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  New concept token = {args.new_concept_token}")
    logger.info(f"  Embedding lr = {args.embedding_lr}")
    logger.info(f"  LoRA lr = {args.learning_rate}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print("Checkpoint not found. Starting fresh.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        # Keep text encoders in eval mode; only embedding layer is live for grads.
        text_encoder_one.eval()
        text_encoder_two.eval()
        # Make sure top-level embedding requires_grad is intact after prepare
        accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)
        accelerator.unwrap_model(text_encoder_two).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, text_encoder_one, text_encoder_two]
            with accelerator.accumulate(models_to_accumulate):
                # ---- Replace [v] placeholder with the registered concept token ----
                raw_prompts = batch["prompts"]
                prompts = [p.replace("[v]", args.new_concept_token) for p in raw_prompts]

                # ---- Tokenize ----
                tokens_one = tokenize_prompt(tokenizer_one, prompts).to(accelerator.device)
                tokens_two = tokenize_prompt(tokenizer_two, prompts).to(accelerator.device)
                tokens_three = tokenize_prompt(tokenizer_three, prompts).to(accelerator.device)

                # ---- Encode: CLIP encoders need gradients; T5 uses no_grad ----
                # CLIP-L and CLIP-G: encode with gradient (embedding params are trainable)
                clip_one_out = text_encoder_one(tokens_one, output_hidden_states=True)
                pooled_one = clip_one_out[0]
                embed_one = clip_one_out.hidden_states[-2].to(dtype=text_encoder_one.dtype)

                clip_two_out = text_encoder_two(tokens_two, output_hidden_states=True)
                pooled_two = clip_two_out[0]
                embed_two = clip_two_out.hidden_states[-2].to(dtype=text_encoder_two.dtype)

                clip_prompt_embeds = torch.cat([embed_one, embed_two], dim=-1)
                pooled_prompt_embeds = torch.cat([pooled_one, pooled_two], dim=-1)

                # T5: no gradient needed
                with torch.no_grad():
                    t5_embeds = _encode_prompt_with_t5(
                        text_encoder_three, tokenizer_three,
                        args.max_sequence_length,
                        prompt=prompts,
                        device=accelerator.device,
                    )

                clip_prompt_embeds = torch.nn.functional.pad(
                    clip_prompt_embeds,
                    (0, t5_embeds.shape[-1] - clip_prompt_embeds.shape[-1]),
                )
                prompt_embeds = torch.cat([clip_prompt_embeds, t5_embeds], dim=-2)

                # ---- VAE encode ----
                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    with torch.no_grad():
                        model_input = vae.encode(pixel_values).latent_dist.sample()

                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # ---- Noise / timesteps ----
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # ---- Transformer forward ----
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                        target.shape[0], -1
                    ),
                    1,
                ).mean()

                # ---- Backward ----
                accelerator.backward(loss)

                # ---- Gradient masking: zero all rows except the new concept token ----
                # This prevents corrupting the original vocabulary embeddings.
                if accelerator.sync_gradients:
                    emb_one = accelerator.unwrap_model(text_encoder_one).get_input_embeddings()
                    emb_two = accelerator.unwrap_model(text_encoder_two).get_input_embeddings()
                    zero_out_non_concept_grads(emb_one, new_token_id_one)
                    zero_out_non_concept_grads(emb_two, new_token_id_two)

                    # Clip LoRA gradients
                    accelerator.clip_grad_norm_(transformer_lora_parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
                                for removing_checkpoint in removing_checkpoints:
                                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "lr_lora": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # ---- Per-epoch validation ----
        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    text_encoder_3=text_encoder_three,
                    transformer=accelerator.unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                # Replace [v] in the validation prompt too
                val_prompt = args.validation_prompt.replace("[v]", args.new_concept_token)
                pipeline_args = {"prompt": val_prompt}
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                )
                del pipeline
                free_memory()

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
