---
layout: post
title: "Transformer Basics: Architecture and Data Flow"
date: 2025-12-13
description: Understanding the transformer architecture from model layers and token sequence perspectives
tags: transformer llm architecture attention deep-learning
categories: technical-notes
giscus_comments: true
related_posts: false
toc:
  sidebar: left
---

Here we review the data flow of transformer architecture. This is a 2-D problem: one from the perspective of model layers; one from the sequence of the tokens.

# Model architecture

In this part, we only look at a simplified version of transformer, namely decoder-only transformer.

<img src="/assets/img/transformer-basics/Screenshot_2025-12-08_at_23.12.31.png" width="60%" alt="Screenshot 2025-12-08 at 23.12.31.png">

## Token embedding

$x_0 = W_E t$

where $t\in \mathbb{R}^{n_{context}\times n_{vocab} }$ is input tokens (one-hot encoded tokens), $W_E\in\mathbb{R}^{d_{model}\times n_{vocab}}$ is the embedding parameter.

Hence $x_0\in\mathbb{R}^{d_{model}\times n_{context}}$

Together ROPE we have this part written in this code format:

```python
# class Qwen3Model forward
				if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        ...

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer()
```

## DecoderLayer

For this part we only focus the multi-head attention and MLP part after it, and skip layer_norm since it does not change the tensor shape, but just normalize it.

In mathematic term, this part essentially does

- Each attention head $h$, the latent state updates according to $x_{i+1} = x_i + \sum_{h\in H_i} h(x_i)$

where $H_i$ is the set of attention heads at layer $i$.

- MLP is run and add the residual: $x_{i+2} = x_{i+1} + m(x_{i+1})$

Here we use the index $i$ to indicate this is calculation on layer $i$. The full picture is simplified as

<img src="/assets/img/transformer-basics/Screenshot_2025-12-09_at_10.49.44.png" width="60%" alt="Screenshot 2025-12-09 at 10.49.44.png">

Here $x_i, x_{i+1}, x_{i+2} \in\mathbb{R}^{n_{context}\times d_{model}}$

Next, let us move on the attention calculation part.

### Attention

The attention (output) is given byU

$$
\mathtt{Attention}(Q,K,V) = \mathtt{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

Usually the softmax part is referred as attention_weight. $\sqrt{d_k}$ is the scaling factor.

The calculation of attention weight can be equivalently expressed in the following equation (less efficient thought)

$$
\mathtt{attnWeight} = \mathtt{softmax}(x^TW_{Q}^T W_K x)
$$

where $W_Q, W_K, W_V \in \mathbb{R}^{n_{head}*d_{head} \times d_{model}}$ (before GQA).

After the calculation of the Attention, there is a projection process to make the $\mathtt{Attention}(Q,K,V)$ to the shape of the $n_{context}\times d_{model}$, so we continue the MLP and residual part of the decoderLayer. Then iterate till the final layer.

### Towards Logits

After the last layer of the decoder, some calculations are needed to derive the final probability distribution of the next token (this part can be different for training/sft).

For inference only (no matter prefill or decoding), we only need the last or latest token hidden_state to predict the next token. In this case we do a slice of the final hidden_states (`logits_to_keep = 1`).

In equations,

- $x_N\in\mathbb{R}^{n_{context}\times d_{model}}$ → $x_{Ns}\in\mathbb{R}^{1\times d_{model}}$
- $\mathtt{logits} = W_u x_{Ns} \in\mathbb{R}^{1\times n_{vocab}}$

where $W_N \in\mathbb{R}^{n_{vocab}\times d_{model}}.$

Then the logits will be normalized as a probability distribution to predict the next token. There are many protocols to play with and accelerate.

# Prefill vs Decode

In the previous sections, we explore and trace through the forward path of the transformer architecture. This can be y-axis of the problem. Then x-axis is from the sequence perspective.

We start with 12 tokens (`<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n`) , and the model generates `'I'm functioning well, thank you for asking!'` (9 new tokens!)

- At the prefill, we run the forward path of the model with 12 tokens, layer by layer
  - this path can be compute bounded (many matrices multiplications)
  - `seq_len = hidden_states.shape[1] = key_states.shape[2] >1`
  - For Qwen3, `past_key_values` is of class `DynamicCache`
  - before we enter a decoderLayer, the `past_key_values` of that layer is not None, but empty, i.e.,

```python
# we only check 0-th layer.

if self.layer_idx == 0:
  seq_len = query_states.shape[2]
  stage = "PREFILL" if seq_len > 1 else "DECODE"

  if past_key_values is not None:
      # Check cache length BEFORE update
      cache_len_before = past_key_values.get_seq_length(self.layer_idx)  # 0


if past_key_values is not None:
  # sin and cos are specific to RoPE models; cache_position needed for the static cache
  cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
  key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

  if self.layer_idx == 0:
      cache_len_after = past_key_values.get_seq_length(self.layer_idx) # 12
```

This can be imagined as drawer opening for a closet, if the drawer is empty, put the KV there and move on to above layer.

- At the decode, the newly generated token from the last step will go through the transformers
  - All the steps before attention calculation can be done by itself (embedding, projection to Q K V etc)
  - Attention is done between this new token to the previous existing tokens.
  - Before we enter a layer, the `past_key_values` of that layer is none empty. For example, we generate one new token after prefill, and now we are at layer 0. The `past_key_values` of this layer has 12 (13) key and values before (after) `past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)`
  - This basically expand the lower-triangle attention matrix (`attn_weights`) by one more row.

# Qwen3 analogies

In this part, we verify the tensor-flow using a minimal inference logic:

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

# =============================================================================
# Load Model and Tokenizer
# =============================================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # torch.bfloat16
    device_map="auto",          # Auto device placement
    trust_remote_code=True,
)

# =============================================================================
# Inference Function
# =============================================================================

def generate(prompt: str, max_new_tokens: int = 10, enable_thinking: bool = False) -> str:
    """Generate a response from the model.

    Args:
        prompt: The user's input
        max_new_tokens: Max tokens to generate
        enable_thinking: If True, model shows reasoning process (default: False)
    """

    # For Qwen3 Thinking models, add /no_think to disable reasoning output

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

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Test prompt
    prompt = "How are you?"

    response = generate(prompt)

    print(f"\n💬 Response:\n{response}")
```

The testing prompt is formatted as `'<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n'` which has 12 tokens.

Some of the key parameters are as follows:

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

Some reference to our notation later

- $d_{model} = 2560 (\mathtt{hidden}\_ \mathtt{size})$
- $n_{vocab} = 151936$

We first focus on the prefilling stage.

### Token embedding

For our testing prompt, the `input_ids` $t$ is of shape `torch.Size([1, 12])` , after embedding the tensor $x_0$ is the `hidden_states` to the first layer of the decoderLayer with shape `torch.Size([1, 12, 2560])` .

The shape of the embedding weights is `torch.Size([151936, 2560])` .

### DecoderLayer-part1

For this part we leave the attention part as a black box for now, and only track the tensor shape of the hidden_states in/out decoderLayer.

This is still for prefill.

The shape is identical to all the layers.

- Input tensor: `hidden_states` is of shape `torch.Size([1, 12, 2560])`
- After Attention + Residual: the shape is unchanged as `torch.Size([1, 12, 2560])`
- After MLP + Residual: still `torch.Size([1, 12, 2560])`

Note here `1` is batch_size.

In code this part is essentially as follows

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

### Attention calculation: part 1

In the real implementation, the big matrix $W^T_QW_K$ is never computed. Instead the tensor $Q,K,V$ are computed

```python
query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
```

where the starting `hidden_states` is of shape `[1, 12, 2560]` and $W_Q, W_K, W_V$ are of shape

`[4096, 2560], [1024, 2560], [1024, 2560]` respectively. In this case, qwen3 use GQA namely every 4 queries share one pair of KV. This saves the KV cache largely.

After the projection, query state ($Q$), key state ($K$), value state ($V$) become

`[1, 12, 4096], [1, 12, 1024], [1, 12, 1024]` . By applying the view, they turn into

`[1, 12, 32, 128], [1, 12, 8, 128], [1, 12, 8, 128]` . Eager and sdpa implementation of attention has the repeat_kv function which duplicates the K V to match Q.

The final transpose(1,2) is to make all tensors into shape `[batch_size, n_heads, seq_len, head_dim]` so that we can calculate the `attn_weights`

```python
attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
```

where attn_weights is of shape `[batch_size, n_heads, seq_len, seq_len]` . This weights matrix is showing: for each head, how much each query token attends to each key tokens. For causal case, the following example show the first head attention matrix

```python
attn_weights[0, 0, :, :] =   # Head 0's attention matrix
       K₀    K₁    K₂    K₃   ...  K₁₁
Q₀   [0.9,  0.0,  0.0,  0.0, ..., 0.0]   # Q₀ only sees K₀ (causal)
Q₁   [0.3,  0.7,  0.0,  0.0, ..., 0.0]   # Q₁ sees K₀, K₁
Q₂   [0.1,  0.2,  0.7,  0.0, ..., 0.0]   # Q₂ sees K₀, K₁, K₂
...
Q₁₁  [0.05, 0.05, 0.1,  0.1, ..., 0.5]   # Q₁₁ sees all K₀-K₁₁
```

Note this is before softmax so we can have the first row with 0.9.

The following code is to add the mask (which makes the unseen part to be -inf for instance), then softmax to normalize it.

```python
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

So far, we have the attention matrix (prefill stage) with shape `[1, 32, 12, 12]` , then the final $\mathtt{Attention}$ is given by $\mathtt{Attn}\_\mathtt{weights} * V$ which is `[1, 32, 12, 128]` .

With a transpose, the final output of the attention function is `attn_output, attn_weights` from `[1, 12, 32, 128], [1, 32, 12, 12]`

Final note about scaling factor def:

```python
self.scaling = self.head_dim**-0.5 # head_dim = 128
```

### Attention calculation: part 2

After calculation of attention, and before we proceed to the MLP part of DecoderLayer, we need to one more projection.

Recall that the final output from the previous attention calculation `attn_output` is of shape `[1, 12, 32, 128]` (i.e., `[batch_size, seq_len, n_heads, head_dim]`).

- First reshape the `attn_output` to `[batch_size, seq_len, n_heads*head_dim]` , i.e., `[1, 12, 4096]`
- Project with weights in shape `[2560, 4096]`
- The final output of the attention block (then we return to the decoderLayer remaining part) is of shape `[1, 12, 2560]`

This part in code is

```python
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = self.o_proj(attn_output)
```

### DecoderLayer-part2

This section is for the remaining part of the decoderLayer: residual, and MLP.

Note that the final output from attention calculation is of the same shape as the input `hidden_states` to decoderLayer, i.e., `[1, 12, 2560]` (`[batch_size, seq_len, d_model]`). For both residual and MLP part, the `hidden_state` will keep this shape, and output.

The input of the previous decoderLayer will be the input to the next one until final one.

The residual + MLP in code is

```python
residual = hidden_states

....# attention cal

hidden_states = residual + hidden_states

# Fully Connected
residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

### Towards Logits

This part is from the output of the last decoderLayer to the logits. The equation in code is given as

```python
hidden_states = outputs.last_hidden_state

# Only compute necessary logits, and do not upcast them to float if we are not computing the loss
slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

logits = self.lm_head(hidden_states[:, slice_indices, :])
```

Here the `hidden_states` is the last layer’s output with shape `[1, 12, 2560]` (`[batch_size, seq_len, d_model]`).

- For both prefill and decode, we only care about the last tokens `hidden_states` , so there is a slice with `logits_to_keep = 1` .
- The sliced `hidden_state` is of shape `[1,1,2560]`
- `lm_head.weight` is of shape `[151936, 2560]` (`[n_vocab, d_model]`)
- Hence the final `logits` is `[1, 1, 151936]` (the first 1 is batch_size)

Note that even though we only care about the last or the latest token’s hidden state, we still need to do the whole prefill because:

1. the last or the latest (prefill or decode) tokens info can not be known without calculating the previous tokens’ info
2. KV cache, so later coming tokens don’t have to recompute the previous tokens key and value to obtain its attention weights

# Ref

https://transformer-circuits.pub/2021/framework/index.html

https://arxiv.org/pdf/1706.03762 (attention is all you need)

https://arxiv.org/pdf/2305.07716 (decoder only transformer)
