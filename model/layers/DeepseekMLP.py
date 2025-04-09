from typing import Optional
import torch.nn as nn

class DeepseekMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: Optional[dict] = None,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                           "Only silu is supported for now.")
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        x = self.down_proj(x)
        return x
