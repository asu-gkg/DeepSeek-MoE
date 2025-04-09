import torch
import torch.nn as nn

class SkipRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, num_experts_per_token, skip_threshold=0.2, skip_enabled=True):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_experts)
        self.num_experts_per_token = num_experts_per_token
        self.skip_threshold = skip_threshold
        self.skip_enabled = skip_enabled

    def forward(self, hidden_states):
        logits = self.linear(hidden_states)
        top_values, top_indices = torch.topk(logits, self.num_experts_per_token, dim=-1)

        if self.skip_enabled:
            mask = (top_values > self.skip_threshold).float()
        else:
            mask = torch.ones_like(top_values)

        return top_indices, mask

    def _copy_parameters(self, old_gate):
        self.linear.weight.data.copy_(old_gate.weight.data)
        if hasattr(old_gate, 'bias') and old_gate.bias is not None:
            self.linear.bias.data.copy_(old_gate.bias.data)
