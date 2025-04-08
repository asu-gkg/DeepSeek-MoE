import torch
import transformers 
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
import torch.nn as nn
# load deepseek-moe-16b-base
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",
    trust_remote_code=True
)



print(f'model: {model}')

config = AutoConfig.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",
    trust_remote_code=True
)

# step1: add skip parameter
config.skip_threshold = 0.2
config.skip_enabled = True

# step2: define SkipLayer, 
class SkipRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_token, skip_thresholad, skip_enabled=True):
        super().__init__()
        # TODO
        