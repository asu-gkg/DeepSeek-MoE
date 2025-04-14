import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Union, Set, Iterable

from model.AutoLoader import AutoWeightsLoader
from model.layers import DeepseekDecoderLayer
from model.layers.DeepseekAttention import DeepseekAttention
from model.layers.DeepseekMLP import DeepseekMLP
from model.layers.DeepseekMoE import DeepseekMoE


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        
    def forward(self, hidden_states, residual=None):
        if residual is not None:
            hidden_states = hidden_states + residual
            
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = hidden_states.to(input_dtype)
        return self.weight * hidden_states, None


class DeepseekModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # self.layers = nn.ModuleList([
        #     DeepseekDecoderLayer(config) for _ in range(config.num_hidden_layers)
        # ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        return_dict=True,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        batch_size, seq_len = inputs_embeds.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        hidden_states = inputs_embeds
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(position_ids, hidden_states, residual)

        hidden_states, _ = self.norm(hidden_states, residual)

        if return_dict:
            return {
                "last_hidden_state": hidden_states,
            }
        return hidden_states


class DeepseekForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model: DeepseekModel = DeepseekModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def load_weights(self, weights: Iterable[Tuple[str,
                                                torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(
            (name, loaded_weight)
            for name, loaded_weight in weights
        )

    def get_input_embeddings(self, input_ids):
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        return_dict=True,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_dict=return_dict,
        )
        
        hidden_states = outputs["last_hidden_state"] if return_dict else outputs
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        if return_dict:
            return {
                "logits": logits,
                "loss": loss
            }
        return logits
        
    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=None,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=True,
        **kwargs
    ):
        max_length = max_length or self.config.max_position_embeddings
        batch_size = input_ids.shape[0]
        
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        for _ in range(input_ids.shape[1], max_length):
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            
            next_token_logits = outputs["logits"][:, -1, :]
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        if token_id < next_token_logits.shape[-1]:
                            next_token_logits[i, token_id] /= repetition_penalty
            
            if do_sample:
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            position_ids = torch.cat([
                position_ids, 
                (position_ids[:, -1] + 1).unsqueeze(-1)
            ], dim=-1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                ], dim=-1)
        
        return input_ids