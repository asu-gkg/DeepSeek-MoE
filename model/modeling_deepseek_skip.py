import torch
import transformers 
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
import torch.nn as nn
from typing import Optional
import sys
import os

# 添加项目根目录到 Python 路径, 这样的话只能在项目的根目录下启动
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.layers.SkipRouter import SkipRouter
from model.layers.DeepseekMLP import DeepseekMLP
from model.layers.DeepseekAttention import DeepseekAttention

# 加载模型以获取 DeepseekMLP
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",
    trust_remote_code=True
)

# 从模型中获取 DeepseekMLP 类
DeepseekMLP = type(model.model.layers[0].mlp)
print(f'DeepseekMLP: {DeepseekMLP}')


print(f'DeepseekMLP: {DeepseekMLP}')

# 递归打印模型所有模块的名字、类型、及结构
def print_model_structure(model, depth=0):
    prefix = "    " * depth
    for name, module in model.named_children():
        print(f"{prefix}{name}: {module.__class__.__name__}")
        print_model_structure(module, depth + 1)

print('model:')
print_model_structure(model)

# print(f'model: {model}')

config = AutoConfig.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-base",
    trust_remote_code=True
)

# step1: add skip parameter
config.skip_threshold = 0.2
config.skip_enabled = True

# step3: SkipLayerMOE
class SkipLayerMOE(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_token, num_shared_experts, intermediate_size, skip_threshold=0.2, skip_enabled=True):
        super().__init__()
        self.router = SkipRouter(hidden_size, num_experts, num_experts_per_token, skip_threshold, skip_enabled)
        self.experts = nn.ModuleList([DeepseekMLP(hidden_size, intermediate_size) for _ in range(num_experts)])
        self.num_experts_per_token = num_experts_per_token
        self.skip_threshold = skip_threshold
        self.skip_enabled = skip_enabled

    def forward(self, hidden_states):
        pass
    
    def _copy_parameters(self, old_moe):
        pass

class SkipLayerDecoderBlock(nn.Module):
    """ 跨层Block: 可以跳过当前层的专家计算 """

    def __init__(self, n_embed, n_head, num_experts, top_k):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = SkipLayerMOE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x, return_skip_info=False):
        # 注意力层总是执行
        x = x + self.sa(self.ln1(x))
        
        # MoE层可能被跳过
        if return_skip_info:
            moe_output, skip_info = self.smoe(self.ln2(x), return_skip_info=True)
            x = x + moe_output
            return x, skip_info
        else:
            x = x + self.smoe(self.ln2(x))
            return x

def replace_moe(model, skip_moe_cls, config):
    for name, module in model.named_modules():
        # 检查是否是 DeepseekMoE
        if 'DeepseekMoE' in str(type(module)):
            # 获取原模块的配置参数
            hidden_size = module.gate.weight.shape[1]  # 输入维度
            num_experts = len(module.experts)  # 专家数量
            intermediate_size = module.experts[0].gate_proj.weight.shape[0]  # 中间维度
            
            # 创建新的 SkipLayerMOE 模块
            new_moe = skip_moe_cls(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_experts_per_token=2,
                num_shared_experts=1,
                intermediate_size=intermediate_size,
                skip_threshold=config.skip_threshold,
                skip_enabled=config.skip_enabled
            )
            
            # 替换模块
            parent_name = name.rsplit('.', 1)[0]
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, name.split('.')[-1], new_moe)
            print(f"Replaced MoE in module: {name}")

# 替换模型中的 MoE
replace_moe(model, SkipLayerMOE, config)

# # 打印替换后的模型结构
# print('model after replace:')
# print_model_structure(model)
