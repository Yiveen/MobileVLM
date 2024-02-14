import os
import copy
import json
import logging
import pathlib
import transformers
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from torch.utils.data import Dataset
from mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, \
    DEFAULT_IM_END_TOKEN
from mobilevlm.train.trainer import VLMTrainer, VIMAVLMTrainer
from mobilevlm import conversation as conversation_lib
from mobilevlm.model.mobilellama import MobileLlamaForCausalLM
from mobilevlm.utils import tokenizer_image_token

import sys
import argparse
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, BitsAndBytesConfig
from mobilevlm.model.vision_encoder import build_vision_tower
from mobilevlm.model.vision_projector import build_vision_projector
from mobilevlm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, \
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from mobilevlm.model.mobilevlm import load_pretrained_model
from mobilevlm.conversation import conv_templates, SeparatorStyle
from mobilevlm.utils import disable_torch_init, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

from .dataset.Lazy_Dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset

local_rank = None
from trl import SFTTrainer

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_type: Optional[str] = field(default='clip')


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize(
        trainer.save_model(output_dir))
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg  # 将所有嵌入的平均值作为新嵌入的数值


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)  # 这里返回了相应的trainer所需要的传入信息

def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()  # 注意几个args
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    disable_torch_init()

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path,
                                              cache_dir=training_args.cache_dir,
                                              model_max_length=training_args.model_max_length,
                                              padding_side="right",use_fast=False) # TODO：需要传入新增的model_path
    model = MobileLlamaForCausalLM.from_pretrained(model_args.model_path, low_cpu_mem_usage=True, **bnb_model_from_pretrained_args)  # 导入模型和tokenizer

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)  # False
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)  # 设置不存在的时候的默认值，并不是设置某个数值 #False
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)  # 都不添加
    model.resize_token_embeddings(len(tokenizer))  # 加入图片的特殊token，给新加入的token提供embedding

    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version] #TODO:现在应该是v1，之后可以根据具体情况修改
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=training_args.device, dtype=compute_dtype) #TODO：是否是冗余的？

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    #TODO:一些不知道哪里用到的配置
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
    model.config.vision_tower_type = training_args.vision_tower_type = model_args.vision_tower_type

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    if training_args.bits in [4, 8]: #是16，不是4，8
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
    # TODO: process prompt and image tensors
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)      #dataloader在这里
    #TODO: 目前不进行generate的方法，所以下面不用，如果需要拿到全部输出的feature去进行后续工作，那么需要下面的stop标准

    # stop_str = conversation_lib.sep.default_conversation if conversation_lib.sep.default_conversation != SeparatorStyle.TWO else conversation_lib.sep.default_conversation.sep2  # '</s>' 什么时候结束输入
    # # stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    #TODO：trainer的修改
    trainer = VIMAVLMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)  # trainer 包含了dataloader
    #TODO：训练过程
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    #TODO: 保存模型
    # model.config.use_cache = True #目前只是输入一次，用不到把前面的输出缓存
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
