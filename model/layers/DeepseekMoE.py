import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from model.layers.DeepseekMLP import DeepseekMLP

class DeepseekMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_token: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        shared_experts: Optional[int] = None,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_token
        self.norm_topk_prob = norm_topk_prob
        
        self.experts = nn.ModuleList([
            DeepseekMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
            )
            for _ in range(num_experts)
        ])
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        self.shared_experts = None
        if shared_experts is not None:
            shared_intermediate_size = intermediate_size * shared_experts
            self.shared_experts = DeepseekMLP(
                hidden_size=hidden_size,
                intermediate_size=shared_intermediate_size,
                hidden_act=hidden_act,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights = torch.softmax(routing_weights, dim=-1)
        
        final_hidden_states = torch.zeros_like(hidden_states)
        
        for expert_idx in range(self.num_experts):
            expert_mask = (selected_experts == expert_idx)
            
            if expert_mask.any():
                expert_input = hidden_states[expert_mask.any(dim=-1)]
                expert_output = self.experts[expert_idx](expert_input)
                expert_weights = routing_weights[expert_mask]
                final_hidden_states[expert_mask.any(dim=-1)] += (
                    expert_output * expert_weights.sum(dim=-1, keepdim=True)
                )
        
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)
            final_hidden_states = final_hidden_states + shared_output
        
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
        return final_hidden_states

    def get_expert_loads(self) -> torch.Tensor:
        with torch.no_grad():
            expert_loads = torch.zeros(self.num_experts, device=self.gate.weight.device)
            for expert_idx in range(self.num_experts):
                expert_loads[expert_idx] = (self.gate.weight[:, expert_idx] != 0).sum()
            return expert_loads
