import argparse
import os
import math
import glob
import inspect
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
from typing import (
    AbstractSet,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist
from torch.optim import AdamW
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tiktoken.load import load_tiktoken_bpe
from moba_naive import moba_attn_varlen_naive
from functools import partial

torch.cuda.empty_cache()

# --- Place it here ---
torch.autograd.set_detect_anomaly(True)

def hf_to_fa(x: torch.Tensor):
    """
    Args:
        x (torch.Tensor): [batch, heads, seqlen, head_dim]

    Returns:
        torch.Tensor: [batch * seqlen, heads, head_dim]
    """
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])


def fa_to_hf(x: torch.Tensor, batch: int):
    """
    Args:
        x (torch.Tensor): [batch * seqlen, heads, head_dim]

    Returns:
        torch.Tensor: [batch, heads, seqlen, head_dim]
    """
    return x.view(batch, -1, x.shape[1], x.shape[2]).permute(0, 2, 1, 3)


# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.hd = config.n_embd // config.n_head
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.hd, bias=False)  # key, query, value projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        if self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_size, config.block_size, config.n_kv_head, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_size, config.block_size, config.n_kv_head, self.hd))

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None, idx=None):
        device = x.device
        self.c_attn = self.c_attn.to(x.device)
        self.c_proj = self.c_proj.to(x.device)
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        input_shape = x.shape[:-1]
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1)
        q, k, v = map(lambda t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        freqs_cis = freqs_cis.to(device)
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        if self.use_kv and not self.training and start_pos >= 0:  # use kv-caching during inference
            self.cache_k = self.cache_k.to(device)
            self.cache_v = self.cache_v.to(device)
            self.cache_k[:B, start_pos : start_pos + T] = k
            self.cache_v[:B, start_pos : start_pos + T] = v
            k = self.cache_k[:B, : start_pos + T]
            v = self.cache_v[:B, : start_pos + T]

        # k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        # v = repeat_kv(v, self.n_rep)

        q, k, v = map(lambda t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        batch, q_heads, q_len, head_dim = q.shape
        _, kv_heads, kv_len, _ = k.shape

        if self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_product_attention(q, k, v, mask == 0 if T > 1 else None)

        # if q_len == kv_len and idx >= 14:
        #     q = hf_to_fa(q)
        #     k = hf_to_fa(k)
        #     v = hf_to_fa(v)
        #     kv_replicas = q_heads // kv_heads
        #     k = torch.repeat_interleave(k, kv_replicas, dim=1)
        #     v = torch.repeat_interleave(v, kv_replicas, dim=1)
        #     cu_seqlens_k = torch.cumsum(
        #         torch.tensor([0] + [kv_len] * batch, device=q.device),
        #         dim=0,
        #         dtype=torch.int32,
        #     )

        #     y = moba_attn_varlen_naive(
        #         q=q,
        #         k=k,
        #         v=v,
        #         cu_seqlens=cu_seqlens_k,
        #         max_seqlen=kv_len,
        #         moba_chunk_size=100,
        #         moba_topk=6
        #     )
        #     y = y.reshape(*input_shape, -1).contiguous()
        # else:
        #     # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
        k = torch.repeat_interleave(k, self.n_rep, dim=1)
        v = torch.repeat_interleave(v, self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
        if mask is not None:
            scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
        att = F.softmax(scores.float(), dim=-1).type_as(q)
        y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None, idx= None):
        x = x + self.attn(self.ln_1(x), freqs_cis, start_pos, mask, idx)
        x = x + self.mlp(self.ln_2(x))
        return x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
class LlamaConfig:
    version: str = "3.1"
    block_size: int = 8192
    vocab_size: int = 128256
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: int = 8
    n_embd: int = 4096
    ffn_dim_multiplier: float = 1.3
    multiple_of: int = 1024
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_scaled_rope: bool = True
    max_gen_batch_size: int = 4
    use_kv: bool = True
    flash: bool = False  # use flashattention?

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        assert self.n_kv_head <= self.n_head
        assert self.n_head % self.n_kv_head == 0
        assert self.n_embd % self.n_head == 0

class LLaMA(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    def forward(self, idx, targets=None, return_logits=True, start_pos=0):
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        if x.device != "cuda:0":
            x = x.to("cuda:0")
        freqs_cis = self.freqs_cis[start_pos:start_pos+t].to(x.device)

        mask = torch.triu(torch.ones((t, t), device=x.device, dtype=torch.bool), diagonal=1)

        for i, block in enumerate(self.transformer.h):
            if i <= 19:
                expected_device = "cuda:0"
            elif 20 <= i <= 25:
                expected_device = "cuda:1"
            else:
                expected_device = "cuda:2"
            
            if x.device != expected_device:
                x = x.to(expected_device)
                freqs_cis = freqs_cis.to(expected_device)
                mask = mask.to(expected_device)
            x = block(x, freqs_cis, start_pos, mask, idx= i)
        if x.device != "cuda:2":
            x = x.to("cuda:2")
        x = self.transformer.ln_f(x)


        if targets is not None:
            logits = self.lm_head(x).float()
            targets = targets.to("cuda:2")
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        else:
            logits = self.lm_head(x[:, [-1], :]).float() 
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    @staticmethod
    def adapt_llama_state_dict_keys(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.pop('tok_embeddings.weight')

        for i in range(config.n_layer):
            for name in ['attention_norm', 'ffn_norm']:
                old_key = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_key = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        for i in range(config.n_layer):
            for name in ['attention.wq', 'attention.wk', 'attention.wv']:
                old_key = f'layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.attn.c_attn.weight'
                if name == 'attention.wq':
                    checkpoint[new_key] = checkpoint.pop(old_key)
                else:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_key] = torch.cat((checkpoint[new_key], checkpoint.pop(old_key)), dim=0)
            old_key = f'layers.{i}.attention.wo.weight'
            new_key = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_key] = checkpoint.pop(old_key)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        for i in range(config.n_layer):
            for name in ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_key = f'layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.pop('output.weight')

        return checkpoint

    @staticmethod
    def adapt_llama_state_dict_keys_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.pop('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        def unpermute(w, n_heads, dim1, dim2):
            return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        for i in range(config.n_layer):
            for name in ['input_layernorm', 'post_attention_layernorm']:
                old_key = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_key = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        for i in range(config.n_layer):
            for name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_key = f'model.layers.{i}.{name}.weight'
                new_key = f'transformer.h.{i}.attn.c_attn.weight'
                if name == 'self_attn.q_proj':
                    checkpoint[new_key] = unpermute(checkpoint.pop(old_key), config.n_head, config.n_embd, config.n_embd)
                else:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.pop(old_key)
                    if name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_head, config.n_kv_head * (config.n_embd // config.n_head), config.n_embd)
                    checkpoint[new_key] = torch.cat((checkpoint[new_key], tensor), dim=0)
            old_key = f'model.layers.{i}.self_attn.o_proj.weight'
            new_key = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_key] = checkpoint.pop(old_key)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        for i in range(config.n_layer):
            for name in ['gate_proj', 'down_proj', 'up_proj']:
                old_key = f'model.layers.{i}.mlp.{name}.weight'
                new_key = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_key] = checkpoint.pop(old_key)

        checkpoint['transformer.ln_f.weight'] = checkpoint.pop('model.norm.weight')

        return checkpoint

    @classmethod
    def from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        assert model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_args = LlamaConfig()

        local_model_path = "/pfs/rdi/cei/home/snpatel/hgface/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
        model = AutoModelForCausalLM.from_pretrained(local_model_path)
        checkpoint = LLaMA.adapt_llama_state_dict_keys_hf(model.state_dict(), model_args)
        #path_to_ckpoint = "/pfs/rdi/cei/home/snpatel/check_cuda/llm.c/model_logs/llama3_SQUAD_MOBADB.pt"
        #checkpoint_SQUAD = torch.load(path_to_ckpoint)
        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        
        #model.load_state_dict(checkpoint_SQUAD['model_state_dict'], strict=False)
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        return model

    @classmethod
    def from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_path):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_args = LlamaConfig()

        ckpt_path = sorted(Path(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        checkpoint = LLaMA.adapt_llama_state_dict_keys(checkpoint, model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=False)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model.tokenizer = tokenizer
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        if zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        return optimizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.

        """
        bsz = len(prompt_tokens)
        assert bsz <= self.config.max_gen_batch_size, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = next(self.parameters()).device

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.block_size, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        total_len = min(self.config.block_size, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        for idx, t in enumerate(prompt_tokens):
            tokens[idx, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = tokens != pad_id

        if min_prompt_len == total_len:
            logits, _ = self.forward(tokens, start_pos=prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens)).to(device)

        for cur_pos in range(min_prompt_len, total_len):
            logits, _ = self.forward(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = next_token.to(tokens.device)
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= ~input_text_mask[:, cur_pos] & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                except ValueError:
                    pass
            out_tokens.append(toks)
        return out_tokens


# -----------------------------------------------------------------------------
# sampling utils

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


class Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
        ]
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            for i in range(self.num_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = num_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        self.eot_id: int = self.special_tokens["<|eot_id|>"]
        self.eom_id: int = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: int = self.special_tokens["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_tokens = [
            self.special_tokens["<|begin_of_text|>"],
            self.special_tokens["<|end_of_text|>"],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        if allowed_special is None:
            allowed_special = set()
        assert type(s) is str

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[int] = []
        for substr in substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

class QADataset(Dataset):
    def __init__(self, data_path: str, tokenizer, block_size: int, split: str = 'train'):
        """
        Args:
            data_path (str): Path to your QA dataset (e.g., JSON file).
            tokenizer: The tokenizer from your Llama model.
            block_size (int): The maximum sequence length for your model.
            split (str): 'train' or 'val' to distinguish data if needed.
        """
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = self._load_data(data_path)
        # if self.tokenizer.pad_token_id is None:
        #     if self.tokenizer.eos_token_id is not None:
        #         self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loaded {len(self.data)} samples for {split} split.")

    def _load_data(self, data_path: str) -> List[Dict]:
        # Implement your data loading logic here.
        # This example assumes a JSON file where each entry is a dict
        # with 'context', 'question', 'answer' keys.
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']

        prompt_text = (
            f"Context: {context}\n"
            f"Question: {question}\n"
        )
        answer_text = f"{answer}"

        # Tokenize the prompt and the answer separately to get their lengths
        prompt_tokens_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False, # We'll add BOS manually at the very start
            truncation=False,         # Don't truncate yet
            return_attention_mask=False,
            return_token_type_ids=False
        ).input_ids

        prompt_tokens_with_bos = [self.tokenizer.bos_token_id] + prompt_tokens_ids

        answer_tokens_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False, # Don't add special tokens here, we'll add EOT manually
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False
        ).input_ids

        answer_tokens_ids = answer_tokens_ids + [self.tokenizer.eos_token_id]

        # Concatenate them for the full sequence that goes into the model
        full_tokens = prompt_tokens_with_bos + answer_tokens_ids

        # Truncate if too long
        if len(full_tokens) > self.block_size:
            full_tokens = full_tokens[:self.block_size]

        # Prepare input_ids (x) and labels (y)
        # For next token prediction, the labels are the input tokens shifted by one.
        # Mask out the prompt tokens in the labels so the loss is only computed on the answer.
        input_ids = torch.tensor(full_tokens, dtype=torch.long)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:] # Shift left by one
        labels[-1] = -100

        # Mask out the prompt part in labels
        # The loss will only be computed for tokens corresponding to the 'answer' part.
        # We set them to -100, which is the default ignore_index for F.cross_entropy.
        labels[:len(prompt_tokens_with_bos) -1] = -100

        # Pad sequences to block_size if they are shorter
        padding_length = self.block_size - len(input_ids)
        if padding_length > 0:
            input_ids = torch.cat([input_ids, torch.full((padding_length,), 128004, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])

        return input_ids, labels

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240801:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    assert header[1] == 7, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240801, "magic number mismatch in the data .bin file"
        assert header[1] == 7, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    for i in range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    assert dtype in {"float32", "bfloat16"}
    version = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_kv_head
    header[7] = model.config.n_embd
    header[8] = model.config.ffn_dim_multiplier
    header[9] = model.config.multiple_of
    header[10] = model.config.norm_eps
    header[11] = model.config.rope_theta
    header[12] = model.config.use_scaled_rope
    header[13] = model.config.max_gen_batch_size
    header[14] = int(model.config.version.split('.')[0]) # major version
    header[15] = int(model.config.version.split('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # now write to file
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = x.size(0) # batch size of the batch, B
    header[2] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    return 2 * (prec * rec) / (prec + rec)

def calculate_mean(data_list):
    if not data_list:  # Handle empty list case to avoid ZeroDivisionError
        return 0
    total_sum = sum(data_list)
    total_count = len(data_list)
    mean = total_sum / total_count
    return mean


if __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hf", type=int, default=1, help="use HuggingFace (default) or use Meta's model")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_train.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="model_logs", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=1, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=2500, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=2500, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=2000, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=20, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=5, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=0, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=0, help="write tensors to disk")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    assert device_type in {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    print(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if (device_type == "cuda") else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # init the model
    if args.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(args.model)
    else:  # use Meta's checkpoint
        assert args.ckpt_dir is not None and os.path.exists(args.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        assert args.tokenizer_path is not None and os.path.exists(args.tokenizer_path), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(args.ckpt_dir, args.tokenizer_path)

    model.train()
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # # Our own version of a simple DistributedDataLoader
    train_dataset = QADataset(data_path='kumar_shivam.json', tokenizer=model.tokenizer, block_size=510, split='train')
    test_dataset = QADataset(data_path='kumar_shivam_val.json', tokenizer=model.tokenizer, block_size=510, split='test')

    with open("kumar_shivam_val.json", "r") as f:
        val_data = json.load(f)

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True, # Shuffle for training
        num_workers=0, # Set to a positive number for faster data loading (e.g., 4)
        pin_memory=True # Speeds up data transfer to GPU
    )

    val_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    # train_loader = DistributedShardedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    # val_loader = None
    # if args.input_val_bin:
    #     val_loader = DistributedShardedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests

    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in bfloat16
        model_to_size = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_size_str = model_to_size[args.model] # e.g. "8B"
        write_model(model, os.path.join(args.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, os.path.join(args.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # init the optimizer
    # optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
    #                                            learning_rate=args.learning_rate, betas=(0.9, 0.95),
    #                                            device_type=device, zero_stage=zero_stage)

    # # # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)


    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode

    print("Placing model stages...")
    for layer in model.transformer.h[:20]:
        layer.to("cuda:0")
    for layer in model.transformer.h[20:26]:
        layer.to("cuda:1")
    for layer in model.transformer.h[26:]:
        layer.to("cuda:2")
    model.lm_head.to("cuda:2")
    model.transformer.wte.to("cuda:0")
    model.transformer.ln_f.to("cuda:2")

    for param in model.transformer.wte.parameters():
        param.requires_grad = False

    N = 14
    for i in range(0, N):
        for param in model.transformer.h[i].parameters():
            param.requires_grad = False

    

    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)
    
    # path_to_ckpoint = "/pfs/rdi/cei/home/snpatel/check_cuda/llm.c/model_logs/llama3_SQUAD_MOBADB.pt"
    # checkpoint_SQUAD = torch.load(path_to_ckpoint)
    # model.load_state_dict(checkpoint_SQUAD['model_state_dict'], strict=False)

    
    # model.eval()

    # exact_matches = []
    # f1_scores = []

    # i = 0

    # for qa in val_data:
    #     context = qa["context"]
    #     question = qa["question"]
    #     answer = qa["answer"]

    #     prompts = []
    #     prompt_text = (
    #         f"Context: {context}\n"
    #         f"Question: {question}\n"
    #     )

    #     prompts.append(prompt_text)
    #     i = i+1
    #     print("generation step", i)

    #     if args.use_hf:
    #         prompt_tokens = [model.tokenizer(x).input_ids for x in prompts]
    #     else:  # Meta
    #         prompt_tokens = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        
    #     generation_tokens = model.generate(prompt_tokens, max_gen_len=50, temperature=0.6, top_p=0.9, echo=False)
    #     results = [{"generation": model.tokenizer.decode(t)} for t in generation_tokens]

    #     for result in results:
    #         gen_ans = result["generation"]

    #     em_score = compute_exact_match(gen_ans, answer)
    #     f1_score = compute_f1(gen_ans, answer)

    #     exact_matches.append(em_score)
    #     f1_scores.append(f1_score)

    # count_of_ones = exact_matches.count(1)
    # mean_f1_score = calculate_mean(f1_scores)

    # print("total data points are", len(f1_scores))
    # print("total number of exact matches", count_of_ones)
    # print("final f1 score", mean_f1_score)


    best_val_loss = float('inf')
    patience = 100 # Number of validation checks to wait for improvement
    no_improvement_count = 0
    min_delta = 1e-4 # Minimum change in validation loss to be considered an improvement

    # --- Model Checkpointing ---
    # This will store the state_dict of the model when a new best_val_loss is achieved
    best_model_state = None
    best_optimizer_state = None
    best_step = 0


    
    # print(model.transformer.h[14].attn.c_attn.in_features)


    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            current_val_loss = 0.0
            # val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    try:
                        x, y = next(iter(val_loader))
                    except StopIteration:
                        val_loader_iter = iter(val_loader)
                        x, y = next(val_loader_iter)
    
                    x = x.to(device)
                    _, loss = model(x, y, return_logits=True)
                    current_val_loss += loss.item()
                current_val_loss  /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {current_val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, current_val_loss))

            if master_process:
                if current_val_loss < best_val_loss - min_delta:
                    best_val_loss = current_val_loss
                    no_improvement_count = 0
                    best_step = step

                    best_checkpoint_filename = "llama3_SQUAD_MOBADB.pt"
                    best_checkpoint_path = os.path.join(args.output_dir, best_checkpoint_filename)

                    # Save the model and optimizer state
                    # Use raw_model.state_dict() as model is likely DDP wrapped
                    torch.save({
                        'model_state_dict': model.state_dict()
                    }, best_checkpoint_path)
                    print0(f"New best validation loss: {best_val_loss:.6f} at step {step}. Model saved.")
                else:
                    no_improvement_count += 1
                    print0(f"Validation loss did not improve. No improvement count: {no_improvement_count}/{patience}")

                should_stop = torch.tensor(no_improvement_count >= patience, dtype=torch.bool).to(device)

            if should_stop.item():
                print0(f"Early stopping triggered at step {step} as validation loss did not improve for {patience} consecutive checks.")
                last_step = True 
                

        # once in a while perform model inference on the master process
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
            and master_process:
            model.eval()
            ans_list = ["185535", "4497329", "816"]
            qa_pairs: List[Dict[str, str]] = [
        {
            "context": """ {
            "465569": "8773427",
            "34041989": "107",
            "43164": "185535",
            "587896": "167",
            "295": "484",
            "2099": "3938281",
            "5787874": "1230",
            "848": "7033",
            "63482": "303",
            "29093127": "295968",
            "242": "508944",
            "52927": "3901760",
            "45490929": "47638",
            "84743787": "74925",
            "42110": "784",
            "8773427": "84245503",
            "98623": "89550",
            "159333": "32031",
            "84245503": "31384",
            "65454": "8494",
            "8748755": "971178",
            "4586589": "804",
            "9713731": "4340",
            "3528": "67967932",
            "2405": "20994772",
            "5132": "91971554",
            "9018": "981169",
            "49669411": "60994023",
            "629467": "7471",
            "6958": "622",
            "6265": "7062",
            "4804250": "80663",
            "359": "87869",
            "6900462": "584",
            "6305683": "101104",
            "20967322": "43164",
            "101104": "530",
            "3993003": "24552",
            "5632014": "71892729",
            "31384": "6305683",
            "842256": "7417555",
            "4678": "6856",
            "60344": "3280",
            "4814399": "4912119",
            "4327": "39194821",
            "13389": "356",
            "43174": "988458",
            "93553454": "4222",
            "93284": "67929905",
            "967872": "6678659",
            "530": "20967322",
            "709": "386",
            "559155": "8772064",
            "35297": "310061",
            "25076907": "2089",
            "101": "8592",
            "46427285": "63403",
            "3224": "33702295",
            "11289091": "100841",
            "5984": "441664" """,
            "question": "What is the final transitive value for key '465569' in the given dictionary?"
        },
        {
            "context": """ "338": "81212",
            "300": "43155",
            "53661": "815052",
            "9514278": "4497329",
            "94916": "6029682",
            "509": "493283",
            "755322": "2054060",
            "680294": "2126825",
            "712": "895",
            "3392": "9514278",
            "33509913": "18767",
            "7374": "740",
            "657": "6033",
            "9070388": "7163",
            "5606": "43183998",
            "53781": "3392",
            "5750139": "21751",
            "865739": "5755",
            "501": "257",
            "63883326": "47363",
            "632244": "7588",
            "336": "36356916",
            "6442": "83739189",
            "782": "6788127",
            "65800": "121",
            "352": "15738",
            "27080": "404973",
            "1648201": "96106085",
            "1810508": "54846",
            "86424890": "219833",
            "702371": "4963592",
            "8042": "569424",
            "31720678": "474",
            "639": "838",
            "10891": "612",
            "54846": "53781",
            "88493504": "18420855",
            "3109": "35246659",
            "248": "8203765",
            "81453045": "14296704",
            "64740": "2376893",
            "6457160": "651902",
            "29267": "85917",
            "15687": "6996092",
            "15738": "52986",
            "7250158": "196",
            "40281": "149332",
            "262": "9886899",
            "196": "206488",
            "965": "662",
            "557188": "971",
            "94031": "154659",
            "859843": "2130",
            "21392": "9411218",
            "6788127": "23748004",
            "56236": "1621",
            "84677": "747",
            "838": "18288611",
            "82429597": "309",
            "429497": "1810508",
            "924731": "31032",
            "52986": "429497",
            "68050891": "8155810",
            "6894811": "4436",
            "989": "67319150",
            "25175": "828143",
            "455902": "22985",
            "126": "2601",
            "2861": "1274",
            "60767": "6481",
            "614941": "56909",
            "752": "9583363",
            "32574862": "556299",
            "4196": "3542",
            "23748004": "352",
            "5218": "120007",
            "7989666": "72490445",
            "53519150": "654",
            "4890539": "68448682",
            "97211165": "465014",
            "864048": "99748",
            "805": "6422" """,
            "question": "What is the final transitive value for key '429497' in the given dictionary?"
        },
        {
            "context": """"9711836": "287",
            "481": "39926930",
            "438357": "20093",
            "184": "67216786",
            "50190549": "8898852",
            "485258": "44029491",
            "5538": "24680",
            "1567669": "4707",
            "355": "21849522",
            "58405": "95158",
            "9848878": "109",
            "643": "5357",
            "471": "2006636",
            "353": "77215523",
            "3220257": "992",
            "304966": "5273288",
            "17065793": "133",
            "5999": "9252041",
            "80708235": "9023",
            "19765991": "30700984",
            "111927": "590",
            "318645": "38038",
            "988": "4310",
            "31022064": "82829716",
            "70844939": "1567843",
            "85461544": "9652061",
            "3674227": "75344819",
            "67012049": "8168",
            "868806": "5332",
            "9266": "1832489",
            "6064": "119",
            "34595": "444",
            "6529353": "3943758",
            "9035": "5741",
            "638666": "79148280",
            "8514": "3120450",
            "656418": "32360",
            "8637": "7190194",
            "385842": "943",
            "6369": "1354416",
            "3943758": "816",
            "9713": "22733",
            "647": "757",
            "594084": "38744147",
            "667378": "19395693",
            "66514046": "6747",
            "28850": "74162768",
            "49222": "3381",
            "4298": "619",
            "15178957": "87360",
            "9620": "326",
            "346": "8637",
            "2886357": "239887",
            "67125950": "9151",
            "103": "9219520",
            "7162": "9997",
            "70149297": "44779560",
            "3709634": "8814",
            "5431700": "64503",
            "4469": "1945979",
            "48672048": "348742",
            "4931": "3796357",
            "3170096": "295",
            "39926930": "34595",
            "632": "36087",
            "510": "19279",
            "7190194": "6529353",
            "66183690": "206784",
            "442526": "4901",
            "27730946": "711653",
            "6957158": "2452",
            "87966465": "424742",
            "256": "960224",
            "97181": "58798239",
            "935151": "2664201",
            "85502102": "5266716",
            "328551": "90130842",
            "30675": "762191",
            "77090225": "680",
            "88328": "61066601",
            "758": "346",
            "9997": "481",
            "403": "995299",
            "38489": "72801",
            "379357": "2824259",
            "15238847": "573083",
            "95098251": "880",
            "81802": "3433061",
            "444": "758",
            "6423": "410",
            "654": "48105309",
            "880351": "2905202" """,
            "question": "What is the final transitive value for key '34595' in the given dictionary?"
        },
    ]
            prompts: List[str] = []
            for qa_pair in qa_pairs:
                context = qa_pair["context"]
                question = qa_pair["question"]
                prompt_text = (
                    f"Context: {context}\n"
                    f"Question: {question}\n"
                )

                prompts.append(prompt_text)
            
            if args.use_hf:
                prompt_tokens = [model.tokenizer(x).input_ids for x in prompts]
            else:  # Meta
                prompt_tokens = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

            generation_tokens = model.generate(prompt_tokens, max_gen_len=15, temperature=0.6, top_p=0.9, echo=False)
            results = [{"generation": model.tokenizer.decode(t)} for t in generation_tokens]
            for prompt, result, ans in zip(prompts, results, ans_list):
                # print(prompt, end="")
                print(f"{result['generation']}")
                print("ans is", ans)
                print("\n==================================\n")
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        # if args.overfit_single_batch:
        #     train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            try:
                # Fetch a batch from the DataLoader
                # This will automatically handle shuffling and batching
                x, y = next(iter(train_loader)) 
            except StopIteration:
                # If an epoch ends, re-create the iterator to start a new epoch
                train_loader_iter = iter(train_loader)
                x, y = next(train_loader_iter)

            x = x.to('cuda:0')
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
                    # forward pass
            out = model.transformer.wte(x)
            freqs_cis = model.freqs_cis[:out.shape[1]]
            mask = torch.triu(
                torch.ones((out.shape[1], out.shape[1]), device=out.device, dtype=torch.bool),
                diagonal=1
            )
           # torch.cuda.empty_cache()

            # Apply checkpointing to blocks of layers in Stage 0
            # You might want to checkpoint each layer individually or groups of layers
            # For simplicity, let's group them or checkpoint per layer if each layer is a checkpointable unit.
            # The `function` passed to checkpoint should take `*args` and return the output.
            # We need to wrap the layer call within a function for checkpointing.

            def create_checkpoint_forward_fn(layer_module):
                # This function acts as the `function` argument for checkpoint.checkpoint
                # It needs to accept all arguments that `layer` would.
                def forward_fn(*inputs):
                    # inputs will contain (out, freqs_cis, 0, mask)
                    # You might need to adjust this based on the exact signature of your layer.forward
                    return layer_module(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
                return forward_fn
            layer_id = 0
            for layer in model.transformer.h[:18]:
                # out = layer(out, freqs_cis, 0, mask) # Original line
                out = checkpoint.checkpoint(create_checkpoint_forward_fn(layer), out, freqs_cis, 0, mask, layer_id, use_reentrant=False)
                layer_id += 1
                # Note: 0 is `start_pos` from your Llama layer definition, `mask` is for attention.
                # Ensure `0` and `mask` are passed as tensors if they are part of autograd graph.
                # If `0` is a constant and `mask` is purely for masking (no gradients flow through it),
                # they might not need to be tracked by checkpoint. However, it's safer to pass them.


            # 3) Stage 1
            out = out.to('cuda:1')
            freqs_cis = freqs_cis.to('cuda:1')
            mask = mask.to('cuda:1')
          #  torch.cuda.empty_cache()
            layer_id = 20
            for layer in model.transformer.h[20:26]:
                # out = layer(out, freqs_cis, 0, mask) # Original line
                out = checkpoint.checkpoint(create_checkpoint_forward_fn(layer), out, freqs_cis, 0, mask, layer_id, use_reentrant=False)
                layer_id += 1

            # 4) Stage 2
            out = out.to('cuda:2')
            freqs_cis = freqs_cis.to('cuda:2')
            mask = mask.to('cuda:2')
         #   torch.cuda.empty_cache()
            layer_id = 26
            for layer in model.transformer.h[26:]:
                # out = layer(out, freqs_cis, 0, mask) # Original line
                out = checkpoint.checkpoint(create_checkpoint_forward_fn(layer), out, freqs_cis, 0, mask, layer_id, use_reentrant=False)
                layer_id += 1

            # Final layers. If ln_f and lm_head are also memory-intensive, you could checkpoint them too,
            # but usually, the transformer blocks are the biggest culprits.
            out = model.transformer.ln_f(out)
            logits = model.lm_head(out)

            # Loss on same device as logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.to(logits.device).view(-1), ignore_index=-100
            )

            # Backward pass
            loss.backward()
            lossf += loss.detach()

            
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # if master_process and not args.inference_only:
    #     checkpoint_path = os.path.join(args.output_dir, "llama3_tinyShakespeare.pt")
    #     torch.save({
    #         'model_state_dict': raw_model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'config': raw_model.config,
    #     }, checkpoint_path)

    if best_checkpoint_path:
       print0(f"Remember: The best model based on validation loss is saved at {best_checkpoint_path} from step {best_step} with loss {best_val_loss:.6f}")

    model.eval()

    exact_matches = []
    f1_scores = []

    i = 0

    for qa in val_data:
        context = qa["context"]
        question = qa["question"]
        answer = qa["answer"]

        prompts = []
        prompt_text = (
            f"Context: {context}\n"
            f"Question: {question}\n"
        )

        prompts.append(prompt_text)
        i = i+1
        print("generation step", i)

        if args.use_hf:
            prompt_tokens = [model.tokenizer(x).input_ids for x in prompts]
        else:  # Meta
            prompt_tokens = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        
        generation_tokens = model.generate(prompt_tokens, max_gen_len=50, temperature=0.6, top_p=0.9, echo=False)
        results = [{"generation": model.tokenizer.decode(t)} for t in generation_tokens]

        for result in results:
            gen_ans = result["generation"]

        em_score = compute_exact_match(gen_ans, answer)
        f1_score = compute_f1(gen_ans, answer)

        exact_matches.append(em_score)
        f1_scores.append(f1_score)

    count_of_ones = exact_matches.count(1)
    mean_f1_score = calculate_mean(f1_scores)

    print("total data points are", len(f1_scores))
    print("total number of exact matches", count_of_ones)
    print("final f1 score", mean_f1_score)


    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
