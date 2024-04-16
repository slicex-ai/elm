# Copyright (c) 2024, SliceX AI, Inc.

import copy
import inspect
import math
import numpy as np
import os

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from elm.utils import *
from elm.positional_embeddings import *


def get_elm_model_map(model_name):
    """Map the model type to corresponding class."""
    elm_model_map = { 
        "rambutan": RambutanSlice,
    }

    return elm_model_map.get(model_name, RambutanSlice)


@dataclass
class ModelArgs:
    """ELM Model Args"""
    model_name_or_path: str = "ELM"
    compile_model: bool = False
    elm_model_class: Optional[str] = "rambutan"
    hidden_size: Optional[int] = 2048
    max_inp_len: Optional[int] = 2048 
    attn_window_size: Optional[int] = max_inp_len 
    num_attention_heads: Optional[int] = 32
    layernorm_eps: float = 1e-5
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    num_layers: Optional[int] = 16
    bits: Optional[int] = 256
    vocab_size: Optional[int] = 50304
    dropout: Optional[int] = 0.1
    use_rotary_embeddings: Optional[bool] = True   
    tokenizer: Optional[str] = None


class ELM(torch.nn.Module):
    """ELM (SliceX GPT) model."""
    def __init__(self,
                 model_args: ModelArgs):
        """Initialize an ELM model instance."""
        super().__init__()

        self.model_args = model_args

        elm_model_class = model_args.elm_model_class
        hidden_size = model_args.hidden_size
        max_inp_len = model_args.max_inp_len
        num_attention_heads = model_args.num_attention_heads
        layernorm_eps = model_args.layernorm_eps
        attention_dropout = model_args.attention_dropout
        hidden_dropout = model_args.hidden_dropout
        num_layers = model_args.num_layers
        bits = model_args.bits
        vocab_size = model_args.vocab_size
        use_rotary_embeddings = model_args.use_rotary_embeddings

        layer_class = get_elm_model_map(elm_model_class)
        
        self.slice_transformer = torch.nn.ModuleDict(dict(
            temb = torch.nn.Embedding(vocab_size, hidden_size),
            pemb = torch.nn.Embedding(max_inp_len, hidden_size) if not use_rotary_embeddings else None,
            drop = torch.nn.Dropout(hidden_dropout),
            h = torch.nn.ModuleList([ layer_class(model_args=model_args) for _ in range(num_layers) ]),
            ln_f = torch.nn.LayerNorm(hidden_size, eps=layernorm_eps),
        ))
                
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        print("Number of model parameters: %.2fM" % (self.get_num_params(False)/1e6,))


    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                targets: Optional[torch.Tensor] = None):
        device = x.device
        batch, seqlen = x.size()

        
        tok_emb = self.slice_transformer.temb(x)

        if not self.model_args.use_rotary_embeddings:
            pos = torch.arange(0, seqlen, dtype=torch.long, device=device)
            pos_emb = self.slice_transformer.pemb(pos)
            x = self.slice_transformer.drop(tok_emb + pos_emb)
        else:
            x = self.slice_transformer.drop(tok_emb)

        ignore_index_id = -100
        loss = torch.zeros(1).to(device)
        loss_denom = 0

        for tlayer in self.slice_transformer.h:
            x = tlayer(x, attention_mask=attention_mask)
            
        x = self.slice_transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            curr_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), 
                                        shift_targets.view(-1),
                                        ignore_index=ignore_index_id)
            loss += curr_loss.float()
            loss_denom += 1
        else:
            logits = self.lm_head(x[:, [-1], :]) 

        loss = loss / loss_denom

        return logits, loss


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), subtract position embeddings if parameter tying applies.
        If there is no parameter sharing, set the flag to False to include parameters for both input/output layers.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.model_args.use_rotary_embeddings:
            n_params -= self.slice_transformer.pemb.weight.numel()
        return n_params


    @torch.no_grad()
    def generate(self, x, max_new_tokens, temperature=0.8, top_k=200, top_p=0.9,
                 return_gen_only=False):
        max_inp_len = self.model_args.max_inp_len

        for _ in range(max_new_tokens):
            x_ctxt = x if x.size(1) <= max_inp_len else x[:, -max_inp_len:]
            
            logits, _ = self(x_ctxt)
            
            next_id = None
            
            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1) 
            else:
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, k = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(logits, dim=-1)

                if top_p is None:
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = sample_top_p(probs, top_p)
            x = torch.cat((x, next_id), dim=1)
    
        if return_gen_only:
            return x[:,-max_new_tokens:]
        
        return x


class RambutanMLP(torch.nn.Module):
    """RambutanMLP version of MLP module used in the ELM (SliceX GPT) Transformer block."""
    def __init__(self, dim=768, bits=32, dropout = 0.0):
        super(RambutanMLP, self).__init__()
        self.dim = dim
        self.bits = bits
      
        self.dropout = torch.nn.Dropout(dropout)

        self.A1_c_w = torch.nn.Linear(self.dim, self.bits, bias=True)

        self.Hexperts = 4
        self.Hexpertemb = torch.nn.Embedding(self.bits, self.dim)
        
        self.expert_aggr = torch.nn.Linear(self.Hexperts, 1)


    def forward(self, x):
        h_c = torch.nn.functional.softmax(self.A1_c_w(x), dim=-1)
                
        v, i = torch.topk(h_c, self.Hexperts)

        if len(x.size()) < 3:
            p = v.unsqueeze(-1).expand(-1,-1,self.dim)
        else:
            p = v.unsqueeze(-1).expand(-1,-1,-1,self.dim)
        
        h_emb = p * self.Hexpertemb(i)

        if len(x.size()) < 3:
            out = self.expert_aggr(h_emb.transpose(1,2)).reshape(h_emb.size(0), -1)
        else:
            out = self.expert_aggr(h_emb.transpose(-2,-1)).reshape(x.size())

        out = x * out 
        out = self.dropout(out)
            
        return out


class RambutanSlice(torch.nn.Module):
    """Rambutan version of ELM (SliceX GPT) Transformer block."""
    def __init__(self,
                 model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        self.num_attention_heads = model_args.num_attention_heads
        self.kv_channels = model_args.hidden_size // model_args.num_attention_heads
        self.ln1 = torch.nn.LayerNorm(model_args.hidden_size, eps=model_args.layernorm_eps)
        self.ln2 = torch.nn.LayerNorm(model_args.hidden_size, eps=model_args.layernorm_eps)
        self.mlp = RambutanMLP(dim=model_args.hidden_size, bits=model_args.bits) 
        self.cattn =  RambutanCausalSelfAttention(model_args=model_args) 


    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor = None):
        res = x
        
        x = self.ln1(x)
        x = self.cattn(x, attention_mask=attention_mask)

        x = res + x
        res = x
        x = self.ln2(x)
        x = self.mlp(x)

        return x + res


class RambutanCausalSelfAttention(torch.nn.Module):
    """Rambutan version of self-attention module used in the ELM (SliceX GPT) transformer block."""

    def __init__(self,
                 model_args: ModelArgs):
        super().__init__()

        self.model_args = model_args

        n_embd = model_args.hidden_size
        n_head = model_args.num_attention_heads
        bias = False 
        dropout = model_args.attention_dropout

        assert n_embd % n_head == 0

        self.c_attn = torch.nn.Linear(n_embd, 3 * n_embd, bias=bias)

        self.c_proj = torch.nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.rotary_embeddings = (
            RotaryEmbedding(n_embd // n_head) if model_args.use_rotary_embeddings else None
        )


    def forward(self, x, attention_mask: torch.Tensor = None):
        B, T, C = x.size()
        device = x.device

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.rotary_embeddings:
            q, k = self.rotary_embeddings(q=q, k=k)
        
        is_causal = True
        attn_mask = None

        if attention_mask is not None:
            att_mask_input = attention_mask 
            att_mask_input = att_mask_input.unsqueeze(-1).expand(B, T, T)

            if is_causal:
                att_mask_causal = torch.tril(torch.ones(T, T)).view(1,T,T).expand(B,T,T).to(device) 
                attn_mask = (att_mask_causal * att_mask_input)
            else:
                attn_mask = att_mask_input
            
            attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_head, T, T)
            attn_mask.float().to(device)

       
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            
            if is_causal and attn_mask is None:
                attn_mask = torch.tril(torch.ones(T, T)).view(1,T,T).expand(B,T,T).to(device) 
                attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_head, T, T)

            if attn_mask is not None:
                att = att.masked_fill(attn_mask == 0, torch.finfo(att.dtype).min)

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        y = self.resid_dropout(self.c_proj(y))

        return y


def init_elm_model(model_args=ModelArgs(), device="cuda", model_config_dict=None):
    """Initialize ELM model using default or model_config parameters."""
    if model_config_dict:
        model_args = ModelArgs(**model_config_dict)

    dtype = torch.bfloat16 if device=="cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = ELM(model_args=model_args).to(dtype=dtype)

    return model

def get_h_layers_in_ckpt(ckpt_state_dict, 
                         layer_name_template = 'slice_transformer.h.{layer_num}.'):
    num_layers_in_ckpt = 0
    from collections import defaultdict
    layer_wise_dict = defaultdict(lambda: defaultdict(list))
    
    layer_num_found = True
    while layer_num_found:
        layer_num_found = False
        for layer_name in ckpt_state_dict.keys():
            if layer_name_template.format(layer_num=num_layers_in_ckpt) in layer_name:
                layer_wise_dict[num_layers_in_ckpt][layer_name] = ckpt_state_dict[layer_name]
                layer_num_found = True
        num_layers_in_ckpt += 1
    return layer_wise_dict

def load_elm_model_from_ckpt(ckpt_dir, device='cuda', load_partial=False, model_args=ModelArgs(), get_num_layers_from_ckpt=True):
    """Load ELM model from local checkpoint."""
    print(f"Loading ELM checkpoint from {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)

    if get_num_layers_from_ckpt:
        layer_name_template = 'slice_transformer.h.{layer_num}.'
        ckpt_layer_wise_dict = get_h_layers_in_ckpt(checkpoint['model'], 
                                                    layer_name_template = layer_name_template)
        model_args.num_layers = len(ckpt_layer_wise_dict)
    model = init_elm_model(model_args=model_args, device=device)
    ckpt_state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k,v in list(ckpt_state_dict.items()):
        if k.startswith(unwanted_prefix):
            ckpt_state_dict[k[len(unwanted_prefix):]] = ckpt_state_dict.pop(k)

    if load_partial:
        mod_state_dict = model.state_dict()
        for k,v in list(ckpt_state_dict.items()):
            if k in mod_state_dict:
                v_size = v.size()
                mod_size = mod_state_dict[k].size()

                if v_size == mod_size:
                    mod_state_dict[k] = v
                else:
                    if len(v_size) == 1:
                        mod_state_dict[k][:v_size[-1]] = v
                    elif len(v_size) == 2:
                        mod_state_dict[k][:v_size[-2], :v_size[-1]] = v

        ckpt_state_dict = mod_state_dict
    load_status = model.load_state_dict(ckpt_state_dict)
    print(load_status)
    model.to(device)

    return model


def sample_top_p(probs, threshold):
    """Perform top-p sampling on probability distribution using a threshold."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > threshold
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token
