import torch.nn as nn

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
