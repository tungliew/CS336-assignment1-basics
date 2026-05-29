# resource accounting
# compute and memory


import math

# matrix-matrix product rule
# Given 𝐴 ∈ ℝ𝑚×𝑛 and 𝐵 ∈ ℝ𝑛×𝑝, the matrix-matrix product 𝐴𝐵 requires 2𝑚𝑛𝑝 FLOPs.
def matmul_flops(m, n, p):
    return 2 * m * n * p


def transformer_stats(vocab_size, context_length, num_layers, d_model, num_heads, d_ff, bytes_per_param=4,):

    assert d_model % num_heads == 0

    head_dim = d_model // num_heads

    # =========================================================
    # PARAMETER COUNT
    # =========================================================
    
    # Token embeddings
    token_embedding_params = vocab_size * d_model

    # Attention
    qkv_params = 3 * (d_model * d_model)
    attention_out_params = d_model * d_model

    attention_params_per_layer = (qkv_params + attention_out_params)

    # Feedforward
    # SwiGLU: w1: d_model -> d_ff, w2: d_ff -> d_model, w3: d_model -> d_ff

    ffn_params_per_layer = (d_model * d_ff + d_ff * d_model + d_model * d_ff)

    # RMSNorm
    rmsnorm_params_per_layer = 2 * d_model

    # Total per transformer layer
    params_per_layer = (attention_params_per_layer + ffn_params_per_layer + rmsnorm_params_per_layer)

    # All transformer layers
    transformer_params = num_layers * params_per_layer

    # Final RMSNorm
    final_norm_params = d_model

    # LM head
    lm_head_params = d_model * vocab_size

    # Total params
    total_params = (token_embedding_params + transformer_params + final_norm_params + lm_head_params)

    # =========================================================
    # MEMORY
    # =========================================================

    param_bytes = total_params * bytes_per_param
    grad_bytes = param_bytes
    optim_bytes = param_bytes * 2  # for AdamW
    memory_bytes = param_bytes + grad_bytes + optim_bytes

    memory_mb = memory_bytes / (1024 ** 2)
    memory_gb = memory_bytes / (1024 ** 3)

    # =========================================================
    # FLOPs
    # =========================================================

    seq_len = context_length

    # Q projection
    q_flops = matmul_flops(seq_len, d_model, d_model,)

    # K projection
    k_flops = matmul_flops(seq_len, d_model, d_model,)

    # V projection
    v_flops = matmul_flops(seq_len, d_model, d_model,)

    # Attention scores: QK^T
    attention_scores_flops = (num_heads * matmul_flops(seq_len, head_dim, seq_len,))

    # Attention weighted sum
    attention_reduce_flops = (num_heads * matmul_flops(seq_len, seq_len, head_dim,))

    # Output projection
    out_proj_flops = matmul_flops(seq_len, d_model, d_model,)

    # FFN
    w1_flops = matmul_flops(seq_len, d_model, d_ff,)

    w2_flops = matmul_flops(seq_len, d_ff, d_model,)

    w3_flops = matmul_flops(seq_len, d_model, d_ff,)

    # Per-layer FLOPs
    flops_per_layer = (q_flops + k_flops + v_flops + attention_scores_flops + attention_reduce_flops + out_proj_flops + w1_flops + w2_flops + w3_flops)

    # All transformer layers
    transformer_flops = num_layers * flops_per_layer

    # LM head
    lm_head_flops = matmul_flops(seq_len, d_model, vocab_size,)

    # Total FLOPs
    total_forward_flops = (transformer_flops + lm_head_flops)

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    print("=" * 80)
    print("MODEL CONFIG")
    print("=" * 80)

    print(f"vocab_size      : {vocab_size:,}")
    print(f"context_length  : {context_length:,}")
    print(f"num_layers      : {num_layers:,}")
    print(f"d_model         : {d_model:,}")
    print(f"num_heads       : {num_heads:,}")
    print(f"d_ff            : {d_ff:,}")

    print("\n")
    print("=" * 80)
    print("PARAMETERS")
    print("=" * 80)

    print(f"Total parameters: {total_params:,}")

    print("\n")
    print("=" * 80)
    print("MEMORY")
    print("=" * 80)

    print(f"Memory (bytes): {memory_bytes:,}")
    print(f"Memory (MB):    {memory_mb:,.2f}")
    print(f"Memory (GB):    {memory_gb:,.2f}")

    print("\n")
    print("=" * 80)
    print("FORWARD PASS FLOPs")
    print("=" * 80)

    print(f"FLOPs per layer: {flops_per_layer:,}")
    print(f"Transformer FLOPs: {transformer_flops:,}")
    print(f"LM head FLOPs: {lm_head_flops:,}")

    print("\n")
    print(f"Total forward FLOPs: {total_forward_flops:,}")
    print(f"Total TFLOPs: {total_forward_flops / 1e12:.3f}")

    return {
        "total_parameters": total_params,
        "memory_bytes": memory_bytes,
        "memory_mb": memory_mb,
        "memory_gb": memory_gb,
        "total_forward_flops": total_forward_flops,
    }


# =============================================================
# EXAMPLE
# =============================================================

transformer_stats(
    vocab_size=50257,
    context_length=256,
    num_layers=4,
    d_model=512,
    num_heads=16,
    d_ff=1344,
)









'''
================================================================================
MODEL CONFIG
================================================================================
vocab_size      : 10,000
context_length  : 1,024
num_layers      : 48
d_model         : 1,600
num_heads       : 25
d_ff            : 4,288


================================================================================
PARAMETERS
================================================================================
Total parameters: 1,511,630,400


================================================================================
MEMORY
================================================================================
Memory (bytes): 6,046,521,600
Memory (MB):    5,766.41
Memory (GB):    5.63


================================================================================
FORWARD PASS FLOPs
================================================================================
FLOPs per layer: 69,835,161,600
Transformer FLOPs: 3,352,087,756,800
LM head FLOPs: 32,768,000,000


Total forward FLOPs: 3,384,855,756,800
Total TFLOPs: 3.385
'''
