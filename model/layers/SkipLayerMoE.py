import torch.nn as nn
from model.layers.SkipRouter import SkipRouter
from model.layers.DeepseekMLP import DeepseekMLP

class SkipLayerMOE(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_token, shared_experts, intermediate_size, skip_threshold=0.2, skip_enabled=True):
        super().__init__()
        self.router = SkipRouter(hidden_size, num_experts, num_experts_per_token, skip_threshold, skip_enabled)
        self.experts = nn.ModuleList([DeepseekMLP(hidden_size, intermediate_size) for _ in range(num_experts)])
        self.num_experts_per_token = num_experts_per_token
        self.skip_threshold = skip_threshold
        self.skip_enabled = skip_enabled
        self.shared_experts = shared_experts

    def forward(self, hidden_states):
        pass
    
    def _copy_parameters(self, old_moe):
        pass