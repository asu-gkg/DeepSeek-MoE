import torch.nn as nn

class DeepseekDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # 注意力模块
        self.self_attn = DeepseekAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings
        )
        
        # 根据配置选择MLP或MoE
        if getattr(config, "moe_enabled", False):
            self.mlp = DeepseekMoE(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_token,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                shared_experts=getattr(config, "shared_experts", None),
                norm_topk_prob=getattr(config, "norm_topk_prob", True)
            )
        else:
            self.mlp = DeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act
            )
        
        # 层归一化
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, positions, hidden_states, residual=None, kv_cache=None):
        # Self Attention
        norm_x, residual = self.input_layernorm(hidden_states, residual)
        attn_output = self.self_attn(positions, norm_x)
        
        # Add output to residual
        hidden_states = hidden_states + attn_output
        
        # MLP/MoE
        norm_x, residual = self.post_attention_layernorm(hidden_states, residual)
        mlp_output = self.mlp(norm_x)
        
        # Add output to residual
        hidden_states = hidden_states + mlp_output
        
        return hidden_states, residual
