When we load qwen-3.5 for full fine-tuning in mt_model.py and select the float32 data type via `model_dtype = torch.float32 if finetune_mode == "full" else compute_dtype`,
qwen-3.5 does not end up actually using float32. This has been experimentally confirmed.

Output of: python tools/inspect_model.py Qwen/Qwen3.5-4B
[transformers] The fast path is not available because one of the required library is not installed. Falling back to torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and https://github.com/Dao-AILab/causal-conv1d

====================================================================================================
1. Model: Qwen/Qwen3.5-4B
====================================================================================================
Qwen3_5ForCausalLM(
  (model): Qwen3_5TextModel(
    (embed_tokens): Embedding(248320, 2560)
    (layers): ModuleList(
      (0-2): 3 x Qwen3_5DecoderLayer(
        (linear_attn): Qwen3_5GatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (norm): Qwen3_5RMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (in_proj_qkv): Linear(in_features=2560, out_features=8192, bias=False)
          (in_proj_z): Linear(in_features=2560, out_features=4096, bias=False)
          (in_proj_b): Linear(in_features=2560, out_features=32, bias=False)
          (in_proj_a): Linear(in_features=2560, out_features=32, bias=False)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (3): Qwen3_5DecoderLayer(
        (self_attn): Qwen3_5Attention(
          (q_proj): Linear(in_features=2560, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (4-6): 3 x Qwen3_5DecoderLayer(
        (linear_attn): Qwen3_5GatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (norm): Qwen3_5RMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (in_proj_qkv): Linear(in_features=2560, out_features=8192, bias=False)
          (in_proj_z): Linear(in_features=2560, out_features=4096, bias=False)
          (in_proj_b): Linear(in_features=2560, out_features=32, bias=False)
          (in_proj_a): Linear(in_features=2560, out_features=32, bias=False)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (7): Qwen3_5DecoderLayer(
        (self_attn): Qwen3_5Attention(
          (q_proj): Linear(in_features=2560, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (8-10): 3 x Qwen3_5DecoderLayer(
        (linear_attn): Qwen3_5GatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (norm): Qwen3_5RMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (in_proj_qkv): Linear(in_features=2560, out_features=8192, bias=False)
          (in_proj_z): Linear(in_features=2560, out_features=4096, bias=False)
          (in_proj_b): Linear(in_features=2560, out_features=32, bias=False)
          (in_proj_a): Linear(in_features=2560, out_features=32, bias=False)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (11): Qwen3_5DecoderLayer(
        (self_attn): Qwen3_5Attention(
          (q_proj): Linear(in_features=2560, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (12-14): 3 x Qwen3_5DecoderLayer(
        (linear_attn): Qwen3_5GatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (norm): Qwen3_5RMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (in_proj_qkv): Linear(in_features=2560, out_features=8192, bias=False)
          (in_proj_z): Linear(in_features=2560, out_features=4096, bias=False)
          (in_proj_b): Linear(in_features=2560, out_features=32, bias=False)
          (in_proj_a): Linear(in_features=2560, out_features=32, bias=False)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (15): Qwen3_5DecoderLayer(
        (self_attn): Qwen3_5Attention(
          (q_proj): Linear(in_features=2560, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (16-18): 3 x Qwen3_5DecoderLayer(
        (linear_attn): Qwen3_5GatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (norm): Qwen3_5RMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (in_proj_qkv): Linear(in_features=2560, out_features=8192, bias=False)
          (in_proj_z): Linear(in_features=2560, out_features=4096, bias=False)
          (in_proj_b): Linear(in_features=2560, out_features=32, bias=False)
          (in_proj_a): Linear(in_features=2560, out_features=32, bias=False)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (19): Qwen3_5DecoderLayer(
        (self_attn): Qwen3_5Attention(
          (q_proj): Linear(in_features=2560, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (20-22): 3 x Qwen3_5DecoderLayer(
        (linear_attn): Qwen3_5GatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (norm): Qwen3_5RMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (in_proj_qkv): Linear(in_features=2560, out_features=8192, bias=False)
          (in_proj_z): Linear(in_features=2560, out_features=4096, bias=False)
          (in_proj_b): Linear(in_features=2560, out_features=32, bias=False)
          (in_proj_a): Linear(in_features=2560, out_features=32, bias=False)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (23): Qwen3_5DecoderLayer(
        (self_attn): Qwen3_5Attention(
          (q_proj): Linear(in_features=2560, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (24-26): 3 x Qwen3_5DecoderLayer(
        (linear_attn): Qwen3_5GatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (norm): Qwen3_5RMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (in_proj_qkv): Linear(in_features=2560, out_features=8192, bias=False)
          (in_proj_z): Linear(in_features=2560, out_features=4096, bias=False)
          (in_proj_b): Linear(in_features=2560, out_features=32, bias=False)
          (in_proj_a): Linear(in_features=2560, out_features=32, bias=False)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (27): Qwen3_5DecoderLayer(
        (self_attn): Qwen3_5Attention(
          (q_proj): Linear(in_features=2560, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (28-30): 3 x Qwen3_5DecoderLayer(
        (linear_attn): Qwen3_5GatedDeltaNet(
          (act): SiLUActivation()
          (conv1d): Conv1d(8192, 8192, kernel_size=(4,), stride=(1,), padding=(3,), groups=8192, bias=False)
          (norm): Qwen3_5RMSNormGated()
          (out_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (in_proj_qkv): Linear(in_features=2560, out_features=8192, bias=False)
          (in_proj_z): Linear(in_features=2560, out_features=4096, bias=False)
          (in_proj_b): Linear(in_features=2560, out_features=32, bias=False)
          (in_proj_a): Linear(in_features=2560, out_features=32, bias=False)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
      (31): Qwen3_5DecoderLayer(
        (self_attn): Qwen3_5Attention(
          (q_proj): Linear(in_features=2560, out_features=8192, bias=False)
          (k_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (v_proj): Linear(in_features=2560, out_features=1024, bias=False)
          (o_proj): Linear(in_features=4096, out_features=2560, bias=False)
          (q_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
          (k_norm): Qwen3_5RMSNorm((256,), eps=1e-06)
        )
        (mlp): Qwen3_5MLP(
          (gate_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (up_proj): Linear(in_features=2560, out_features=9216, bias=False)
          (down_proj): Linear(in_features=9216, out_features=2560, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
        (post_attention_layernorm): Qwen3_5RMSNorm((2560,), eps=1e-06)
      )
    )
    (norm): Qwen3_5RMSNorm((2560,), eps=1e-06)
    (rotary_emb): Qwen3_5TextRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2560, out_features=248320, bias=False)
)

====================================================================================================
2. Module types (count)
====================================================================================================
  Linear                                      249
  Qwen3_5RMSNorm                               81
  SiLUActivation                               56
  Qwen3_5DecoderLayer                          32
  Qwen3_5MLP                                   32
  Qwen3_5GatedDeltaNet                         24
  Conv1d                                       24
  Qwen3_5RMSNormGated                          24
  Qwen3_5Attention                              8
  Qwen3_5ForCausalLM                            1
  Qwen3_5TextModel                              1
  Embedding                                     1
  ModuleList                                    1
  Qwen3_5TextRotaryEmbedding                    1

====================================================================================================
3. LoRA candidates: Linear modules by name (what peft.lora.target_modules matches)
====================================================================================================
  name             count  in -> out          parent                      adapter params at r=8
  out_proj            24  4096 -> 2560       Qwen3_5GatedDeltaNet                    1,277,952
  in_proj_qkv         24  2560 -> 8192       Qwen3_5GatedDeltaNet                    2,064,384
  in_proj_z           24  2560 -> 4096       Qwen3_5GatedDeltaNet                    1,277,952
  in_proj_b           24  2560 -> 32         Qwen3_5GatedDeltaNet                      497,664
  in_proj_a           24  2560 -> 32         Qwen3_5GatedDeltaNet                      497,664
  gate_proj           32  2560 -> 9216       Qwen3_5MLP                              3,014,656
  up_proj             32  2560 -> 9216       Qwen3_5MLP                              3,014,656
  down_proj           32  9216 -> 2560       Qwen3_5MLP                              3,014,656
  q_proj               8  2560 -> 8192       Qwen3_5Attention                          688,128
  k_proj               8  2560 -> 1024       Qwen3_5Attention                          229,376
  v_proj               8  2560 -> 1024       Qwen3_5Attention                          229,376
  o_proj               8  4096 -> 2560       Qwen3_5Attention                          425,984
  lm_head              1  2560 -> 248320                                             2,007,040

  Ready to paste into conf/peft/<option>.yaml (lora: block) or as an override:
    attention only:            target_modules: [q_proj, k_proj, v_proj, o_proj]
    all linear except lm_head: target_modules: [out_proj, in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, gate_proj, up_proj, down_proj, q_proj, k_proj, v_proj, o_proj]
  PEFT's built-in default for model_type=qwen3_5: none (for information; hfmt always uses the listed target_modules)

====================================================================================================
4. Parameters
====================================================================================================
  total                            4,205,751,296
  input embedding                    635,699,200
  lm_head                                      0  (tied to the input embedding)
  rest (layers, norms)             3,570,052,096
  of which Linear weights (P_lin, quantized by qlora): 3,569,090,560

====================================================================================================
5. Training memory estimate (peak *allocated* GPU memory for one training step)
====================================================================================================
  A guide, not a guarantee (expect +-30%); gradient checkpointing on; AdamW; bf16 activations.
  The GPU needs more: + ~1.5 GiB CUDA context, + what the caching allocator keeps
  (nvidia-smi / max_memory_reserved; with varying batch shapes it can grow to fill the GPU).
  Checked on A100s (2026-10-18; Qwen2.5-0.5B/1.5B, tiny-aya; 20 steps): measured / estimate = 0.86-1.04.
  Values: P=4,205,751,296  P_lin=3,569,090,560  h=2560  I=9216  L=32  n_kv*head_dim=1024  V=248,320

  full at batch 16 x length 512  (T = trainable = 4,205,751,296)
    weights (fp32)                             4*P = 4*4,205,751,296                                                                            =   15.67 GiB
    gradients (fp32)                           4*T = 4*4,205,751,296                                                                            =   15.67 GiB
    AdamW state (m, v fp32)                    8*T = 8*4,205,751,296                                                                            =   31.34 GiB
    activations saved (layer inputs)           2*b*s*h*L = 2*16*512*2560*32                                                                     =    1.25 GiB
    one layer recomputed in backward           2*b*s*(6h + 2*kv + 4I) = 2*16*512*(6*2560 + 2*1024 + 4*9216)                                     =    0.83 GiB
    logits + loss                              14*b*s*V = 14*16*512*248,320                                                                     =   26.52 GiB
    total                                                                                                                                       =   91.27 GiB

  lora   [q_proj, v_proj] r=8 at batch 16 x length 512  (T = trainable = 917,504)
    weights (bf16) + adapters (fp32)           2*P + 4*T = 2*4,205,751,296 + 4*917,504                                                          =    7.84 GiB
    gradients (fp32)                           4*T = 4*917,504                                                                                  =    0.00 GiB
    AdamW state (m, v fp32)                    8*T = 8*917,504                                                                                  =    0.01 GiB
    activations saved (layer inputs)           2*b*s*h*L = 2*16*512*2560*32                                                                     =    1.25 GiB
    one layer recomputed in backward           2*b*s*(6h + 2*kv + 4I) = 2*16*512*(6*2560 + 2*1024 + 4*9216)                                     =    0.83 GiB
    logits + loss                              14*b*s*V = 14*16*512*248,320                                                                     =   26.52 GiB
    total                                                                                                                                       =   36.45 GiB

  qlora  [q_proj, v_proj] r=8 at batch 16 x length 512  (T = trainable = 917,504)
    weights (NF4 Linear, fp32 rest) + adapters 0.516*P_lin + 4*(P-P_lin) + 4*T = 0.516*3,569,090,560 + 4*636,660,736 + 4*917,504                =    4.09 GiB
    gradients (fp32)                           4*T = 4*917,504                                                                                  =    0.00 GiB
    AdamW state (m, v fp32)                    8*T = 8*917,504                                                                                  =    0.01 GiB
    activations saved (layer inputs)           2*b*s*h*L = 2*16*512*2560*32                                                                     =    1.25 GiB
    one layer recomputed in backward           2*b*s*(6h + 2*kv + 4I) = 2*16*512*(6*2560 + 2*1024 + 4*9216)                                     =    0.83 GiB
    logits + loss                              14*b*s*V = 14*16*512*248,320                                                                     =   26.52 GiB
    total                                                                                                                                       =   32.70 GiB

  lora   all linear r=16 at batch 16 x length 512  (T = trainable = 32,464,896)
    weights (bf16) + adapters (fp32)           2*P + 4*T = 2*4,205,751,296 + 4*32,464,896                                                       =    7.95 GiB
    gradients (fp32)                           4*T = 4*32,464,896                                                                               =    0.12 GiB
    AdamW state (m, v fp32)                    8*T = 8*32,464,896                                                                               =    0.24 GiB
    activations saved (layer inputs)           2*b*s*h*L = 2*16*512*2560*32                                                                     =    1.25 GiB
    one layer recomputed in backward           2*b*s*(6h + 2*kv + 4I) = 2*16*512*(6*2560 + 2*1024 + 4*9216)                                     =    0.83 GiB
    logits + loss                              14*b*s*V = 14*16*512*248,320                                                                     =   26.52 GiB
    total                                                                                                                                       =   36.92 GiB

  Per token in a batch (all modes): 2*h*L + 2*(6h + 2*kv + 4I) + 14*V = 3,748,864 B = 3.58 MiB
  So: allocated ~ fixed part (weights, gradients, AdamW) + batch * length * per-token part.

  GiB allocated                           16x128      16x512      32x512      8x2048
  full                                      69.8        91.3       119.9       119.9
  lora   [q_proj, v_proj] r=8               15.0        36.4        65.1        65.1
  qlora  [q_proj, v_proj] r=8               11.3        32.7        61.3        61.3
  lora   all linear r=16                    15.5        36.9        65.5        65.5
  (columns: batch x sequence length after padding to the longest in the batch)

====================================================================================================
6. Config highlights
====================================================================================================
  model_type             qwen3_5
  architectures          ['Qwen3_5ForConditionalGeneration']
  layers                 32
  hidden size            2560
  intermediate size      9216
  attention heads        16
  key/value heads        4
  head dim               256
  vocabulary             248320
  max positions          262144
  sliding window         None
  dtype                  torch.bfloat16
  tie_word_embeddings    True
  quantization_config    none

====================================================================================================
7. Tokenizer and chat template
====================================================================================================
  pad_token  '<|endoftext|>'                id 248044
  bos_token  None                           id None
  eos_token  '<|im_end|>'                   id 248046
  vocabulary (tokenizer)  248,077
  chat template: yes; hfmt's prompt adds 17 tokens around a one-word source
  (the example instruction, the template's turn markers and any default system prompt; apply_chat_template
  kwargs {'enable_thinking': False}). They count toward decode.max_length and the sequence length above.


config.json:
{
    "architectures": [
        "Qwen3_5ForConditionalGeneration"
    ],
    "image_token_id": 248056,
    "model_type": "qwen3_5",
    "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9216,
        "layer_types": [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention"
        ],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": true,
        "use_cache": true,
        "vocab_size": 248320,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
            "mrope_interleaved": true,
            "mrope_section": [
                11,
                11,
                10
            ],
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25
        }
    },
    "tie_word_embeddings": true,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 248057,
    "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_5",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
    },
    "vision_end_token_id": 248054,
    "vision_start_token_id": 248053
}

We need to experimentally confirm that float32 works.