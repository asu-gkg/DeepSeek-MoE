import torch
import transformers 
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
import torch.nn as nn
from typing import Optional
import sys
import os
from typing import Optional, Iterable, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.layers.SkipLayerMoE import SkipLayerMoE
from model.layers.SkipRouter import SkipRouter
from model.layers.DeepseekMLP import DeepseekMLP
from model.layers.DeepseekAttention import DeepseekAttention
from model.DeepseekModel import DeepseekForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",
    trust_remote_code=True
)
print(f'model.named_parameters(): {model.named_parameters()}')

config = AutoConfig.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",
    trust_remote_code=True
)

weights: Iterable[Tuple[str, torch.Tensor]] = model.named_parameters()
model = DeepseekForCausalLM(config)

model.load_weights(weights)

# # 从模型中获取 DeepseekMLP 类
# DeepseekMLP = type(model.model.layers[0].mlp)
# print(f'DeepseekMLP: {DeepseekMLP}')

# def print_model_structure(model, depth=0):
#     prefix = "    " * depth
#     for name, module in model.named_children():
#         print(f"{prefix}{name}: {module.__class__.__name__}")
#         print_model_structure(module, depth + 1)

# print('model:')
# print_model_structure(model)

# print(f'model: {model}')


# step1: add skip parameter
config.skip_threshold = 0.2
config.skip_enabled = True
