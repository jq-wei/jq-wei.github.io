---
layout: post
title: "Kimi Attention Residual: Understanding Transformer Inference"
date: 2026-03-22 08:00:00
description: A deep dive into transformer inference mechanics, exploring prefill vs decode stages, attention calculations, and KV caching in Qwen3
tags: transformer attention inference kv-cache llm
categories: deep-learning
giscus_comments: true
related_posts: true
toc:
  sidebar: left
---

In this post, we trace through the forward path of transformer architecture, exploring both the model structure (y-axis) and sequence processing (x-axis). We'll examine how transformers handle inference through prefill and decode stages, with concrete examples using Qwen3-4B-Instruct.

## Overview of Transformer Forward Pass

The transformer forward pass can be understood from two perspectives:

1. **Model Architecture (y-axis)**: Layer-by-layer computation through the network
2. **Sequence Processing (x-axis)**: How tokens are processed through prefill and decode stages

### Basic Forward Path

For a language model, the forward path follows this general structure:

1. **Token Embedding**: Convert input token IDs to dense vectors
2. **Decoder Layers**: Process through N transformer layers (each with attention + MLP)
3. **Generate Logits**: Project final hidden states to vocabulary space
4. **Next Token Prediction**: Sample from the probability distribution

Mathematically, for next-token prediction:

- Input: $$x_N\in\mathbb{R}^{n_{context}\times d_{model}}$$
- Output: $$x_{Ns}\in\mathbb{R}^{1\times d_{model}}$$
- Logits: $$\mathtt{logits} = W_u x_{Ns} \in\mathbb{R}^{1\times n_{vocab}}$$

where $$W_N \in\mathbb{R}^{n_{vocab}\times d_{model}}$$.

The logits are then normalized as a probability distribution to predict the next token. There are many protocols to accelerate this process.

## Prefill vs Decode

The transformer inference has two distinct stages from the sequence perspective:

### Prefill Stage

Starting with 12 tokens (`<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n`), the model processes all input tokens simultaneously:

- **Computation**: Layer-by-layer forward pass with all 12 tokens
- **Characteristics**:
  - Compute-bounded (many matrix multiplications)
  - `seq_len = hidden_states.shape[1] = key_states.shape[2] > 1`
  - For Qwen3, `past_key_values` uses `DynamicCache` class
  - Before entering a decoder layer, `past_key_values` for that layer is empty

Example code check at layer 0:

```python
# We only check 0-th layer
if self.layer_idx == 0:
    seq_len = query_states.shape[2]
    stage = "PREFILL" if seq_len > 1 else "DECODE"
    
    if past_key_values is not None:
        # Check cache length BEFORE update
        cache_len_before = past_key_values.get_seq_length(self.layer_idx)  # 0

if past_key_values is not None:
    # sin and cos are specific to RoPE models; cache_position needed for the static cache
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(
        key_states, value_states, self.layer_idx, cache_kwargs
    )
    
    if self.layer_idx == 0:
        cache_len_after = past_key_values.get_seq_length(self.layer_idx)  # 12
```

This can be imagined as opening an empty drawer in a closet - if the drawer is empty, we put the KV pairs there and move on to the next layer.

### Decode Stage

After prefill, each newly generated token goes through the transformer:

- **Per-token Processing**: Each new token is processed individually
- **Attention**: The new token attends to all previous tokens
- **KV Cache**: Before entering a layer, `past_key_values` contains all previous keys and values
  - For example, after generating one token post-prefill at layer 0:
  - Cache has 12 (or 13 after update) key-value pairs
  - This expands the lower-triangle attention matrix by one more row

## Qwen3 Architecture Analysis

Let's verify the tensor flow using minimal inference logic with Qwen3-4B-Instruct:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model path - can be HuggingFace ID or local path
MODEL_ID = "local/path/to/Qwen3-4B-Instruct-2507"

# Device selection for Mac M4
if torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple Silicon GPU
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Inference Function
def generate(prompt: str, max_new_tokens: int = 10, enable_thinking: bool = False) -> str:
    """Generate a response from the model.
    
    Args:
        prompt: The user's input
        max_new_tokens: Max tokens to generate
        enable_thinking: If True, model shows reasoning process (default: False)
    """
    
    # Format as chat (Qwen uses ChatML format)
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(f"Formatted prompt:\n{repr(text)}")
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode (skip input tokens)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    
    return response.strip()

# Main
if __name__ == "__main__":
    prompt = "How are you?"
    response = generate(prompt)
    print(f"\n💬 Response:\n{response}")
```

### Key Model Parameters

```python
"head_dim": 128,
"hidden_size": 2560,
"intermediate_size": 9728,
"max_window_layers": 36,
"num_attention_heads": 32,
"num_hidden_layers": 36,
"num_key_value_heads": 8,
"vocab_size": 151936
```

Notation mapping:
- $$d_{model} = 2560$$ (`hidden_size`)
- $$n_{vocab} = 151936$$

For our test prompt, `input_ids` $$t$$ has shape `torch.Size([1, 12])`.

## Token Embedding

After embedding, the tensor $$x_0$$ becomes the `hidden_states` input to the first decoder layer:

- Shape: `torch.Size([1, 12, 2560])`
- Embedding weights shape: `torch.Size([151936, 2560])`

<img src="{{ '/assets/img/kimi-attention/Screenshot_2026-03-21_at_22.22.05.png' | relative_url }}" alt="Token embedding visualization" width="80%">

## Decoder Layer Structure

### Part 1: High-Level Flow

The decoder layer maintains consistent tensor shapes throughout:

- **Input**: `hidden_states` with shape `[1, 12, 2560]`
- **After Attention + Residual**: `[1, 12, 2560]` (unchanged)
- **After MLP + Residual**: `[1, 12, 2560]` (unchanged)

Note: The first dimension `1` is the batch size.

Code structure:

```python
# class Qwen3DecoderLayer forward
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)

# Self Attention
hidden_states, _ = self.self_attn(...)
hidden_states = residual + hidden_states

# Fully Connected
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

### Part 2: Attention Calculation Details

In practice, the large matrix $$W^T_Q W_K$$ is never computed directly. Instead, we compute $$Q, K, V$$ tensors:

```python
query_states = self.q_norm(
    self.q_proj(hidden_states).view(hidden_shape)
).transpose(1, 2)

key_states = self.k_norm(
    self.k_proj(hidden_states).view(hidden_shape)
).transpose(1, 2)

value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

**Tensor Shapes Evolution**:

Starting `hidden_states`: `[1, 12, 2560]`

Projection weights:
- $$W_Q$$: `[4096, 2560]`
- $$W_K$$: `[1024, 2560]`
- $$W_V$$: `[1024, 2560]`

Qwen3 uses **Grouped Query Attention (GQA)** where every 4 queries share one pair of KV, significantly reducing KV cache size.

After projection:
- Query ($$Q$$): `[1, 12, 4096]`
- Key ($$K$$): `[1, 12, 1024]`
- Value ($$V$$): `[1, 12, 1024]`

After `view()` operation:
- $$Q$$: `[1, 12, 32, 128]`
- $$K$$: `[1, 12, 8, 128]`
- $$V$$: `[1, 12, 8, 128]`

The `transpose(1, 2)` reshapes all to: `[batch_size, n_heads, seq_len, head_dim]`

#### Computing Attention Weights

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
```

Result: `attn_weights` shape is `[batch_size, n_heads, seq_len, seq_len]`

This matrix shows, for each head, how much each query token attends to each key token.

**Example for causal attention** (first head):

```python
attn_weights[0, 0, :, :] =   # Head 0's attention matrix
       K₀    K₁    K₂    K₃   ...  K₁₁
Q₀   [0.9,  0.0,  0.0,  0.0, ..., 0.0]   # Q₀ only sees K₀ (causal)
Q₁   [0.3,  0.7,  0.0,  0.0, ..., 0.0]   # Q₁ sees K₀, K₁
Q₂   [0.1,  0.2,  0.7,  0.0, ..., 0.0]   # Q₂ sees K₀, K₁, K₂
...
Q₁₁  [0.05, 0.05, 0.1,  0.1, ..., 0.5]   # Q₁₁ sees all K₀-K₁₁
```

Note: This is before softmax, so values can exceed 1.0.

#### Applying Causal Mask and Softmax

```python
if attention_mask is not None:
    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
    attn_weights = attn_weights + causal_mask

attn_weights = nn.functional.softmax(
    attn_weights, dim=-1, dtype=torch.float32
).to(query.dtype)
```

**Final attention output**:

$$\mathtt{Attn}_\mathtt{weights} \times V$$

- Attention weights: `[1, 32, 12, 12]` (prefill stage)
- Final output: `[1, 32, 12, 128]`
- After transpose: `[1, 12, 32, 128]`

Returns: `attn_output, attn_weights` with shapes `[1, 12, 32, 128], [1, 32, 12, 12]`

**Scaling factor**:

```python
self.scaling = self.head_dim**-0.5  # head_dim = 128
```

<img src="{{ '/assets/img/kimi-attention/Screenshot_2026-03-21_at_22.24.47.png' | relative_url }}" alt="Attention mechanism visualization" width="80%">

### Part 3: Output Projection

After computing attention, we need one more projection before proceeding to the MLP:

Recall `attn_output` shape: `[1, 12, 32, 128]` (i.e., `[batch_size, seq_len, n_heads, head_dim]`)

1. Reshape to `[batch_size, seq_len, n_heads*head_dim]` → `[1, 12, 4096]`
2. Project with weights `[2560, 4096]`
3. Final output: `[1, 12, 2560]`

Code:

```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.o_proj(attn_output)
```

### Part 4: MLP and Residual Connections

The output from attention has the same shape as the input `hidden_states`: `[1, 12, 2560]` (`[batch_size, seq_len, d_model]`).

Both the residual connection and MLP maintain this shape throughout:

```python
residual = hidden_states

# ... attention calculation ...

hidden_states = residual + hidden_states

# Fully Connected
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

The output of each decoder layer becomes the input to the next, continuing through all 36 layers.

## From Hidden States to Logits

After the final decoder layer, we convert hidden states to logits:

```python
hidden_states = outputs.last_hidden_state

# Only compute necessary logits
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

logits = self.lm_head(hidden_states[:, slice_indices, :])
```

**Key points**:

- Input `hidden_states`: `[1, 12, 2560]` (`[batch_size, seq_len, d_model]`)
- For both prefill and decode, we only care about the **last token's** hidden state
- With `logits_to_keep = 1`, sliced `hidden_state`: `[1, 1, 2560]`
- `lm_head.weight`: `[151936, 2560]` (`[n_vocab, d_model]`)
- Final `logits`: `[1, 1, 151936]` (batch_size=1)

<img src="{{ '/assets/img/kimi-attention/Screenshot_2026-03-22_at_07.31.18.png' | relative_url }}" alt="Logits generation process" width="80%">

### Why Compute All Tokens During Prefill?

Even though we only need the last token's logits, we must compute the entire sequence because:

1. **Context Dependency**: The last token's representation depends on all previous tokens through attention
2. **KV Caching**: We populate the KV cache so future tokens don't need to recompute previous keys and values

This is the fundamental tradeoff in transformer inference - prefill is computationally expensive but enables efficient decode.

## References

- [Transformer Circuits Thread](https://transformer-circuits.pub/2021/framework/index.html)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762) - Original transformer paper
- [Decoder-Only Transformers](https://arxiv.org/pdf/2305.07716) - Analysis of decoder-only architecture
