We want to add a new model: CohereLabs/tiny-aya-global

====================================================================================================
1. Model: CohereLabs/tiny-aya-global
====================================================================================================
Cohere2ForCausalLM(
  (model): Cohere2Model(
    (embed_tokens): Embedding(262144, 2048, padding_idx=0)
    (layers): ModuleList(
      (0-35): 36 x Cohere2DecoderLayer(
        (self_attn): Cohere2Attention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): Cohere2MLP(
          (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): Cohere2LayerNorm()
      )
    )
    (norm): Cohere2LayerNorm()
    (rotary_emb): Cohere2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=262144, bias=False)
)

====================================================================================================
2. Module types (count)
====================================================================================================
  Linear                                      253
  Cohere2LayerNorm                             37
  Cohere2DecoderLayer                          36
  Cohere2Attention                             36
  Cohere2MLP                                   36
  SiLUActivation                               36
  Cohere2ForCausalLM                            1
  Cohere2Model                                  1
  Embedding                                     1
  ModuleList                                    1
  Cohere2RotaryEmbedding                        1

====================================================================================================
3. LoRA candidates: Linear modules by name (what peft.lora.target_modules matches)
====================================================================================================
  name             count  in -> out          parent                      adapter params at r=8
  q_proj              36  2048 -> 2048       Cohere2Attention                        1,179,648
  k_proj              36  2048 -> 512        Cohere2Attention                          737,280
  v_proj              36  2048 -> 512        Cohere2Attention                          737,280
  o_proj              36  2048 -> 2048       Cohere2Attention                        1,179,648
  gate_proj           36  2048 -> 11008      Cohere2MLP                              3,760,128
  up_proj             36  2048 -> 11008      Cohere2MLP                              3,760,128
  down_proj           36  11008 -> 2048      Cohere2MLP                              3,760,128
  lm_head              1  2048 -> 262144                                             2,113,536

  Ready to paste into conf/peft/<option>.yaml (lora: block) or as an override:
    attention only:            target_modules: [q_proj, k_proj, v_proj, o_proj]
    all linear except lm_head: target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  PEFT's built-in default for model_type=cohere2: none (for information; hfmt always uses the listed target_modules)

====================================================================================================
4. Parameters
====================================================================================================
  total                            3,349,227,520
  input embedding                    536,870,912
  lm_head                                      0  (tied to the input embedding)
  rest (layers, norms)             2,812,356,608
  of which Linear weights (P_lin, quantized by qlora): 2,812,280,832

====================================================================================================
5. Training memory estimate (peak *allocated* GPU memory for one training step)
====================================================================================================
  A guide, not a guarantee (expect +-30%); gradient checkpointing on; AdamW; bf16 activations.
  The GPU needs more: + ~1.5 GiB CUDA context, + what the caching allocator keeps
  (nvidia-smi / max_memory_reserved; with varying batch shapes it can grow to fill the GPU).
  Checked on A100s (2026-10-18; Qwen2.5-0.5B/1.5B, tiny-aya; 20 steps): measured / estimate = 0.86-1.04.
  Values: P=3,349,227,520  P_lin=2,812,280,832  h=2048  I=11008  L=36  n_kv*head_dim=512  V=262,144

  full at batch 16 x length 512  (T = trainable = 3,349,227,520)
    weights (fp32)                             4*P = 4*3,349,227,520                                                                            =   12.48 GiB
    gradients (fp32)                           4*T = 4*3,349,227,520                                                                            =   12.48 GiB
    AdamW state (m, v fp32)                    8*T = 8*3,349,227,520                                                                            =   24.95 GiB
    activations saved (layer inputs)           2*b*s*h*L = 2*16*512*2048*36                                                                     =    1.12 GiB
    one layer recomputed in backward           2*b*s*(6h + 2*kv + 4I) = 2*16*512*(6*2048 + 2*512 + 4*11008)                                     =    0.88 GiB
    logits + loss                              14*b*s*V = 14*16*512*262,144                                                                     =   28.00 GiB
    total                                                                                                                                       =   79.91 GiB

  lora   [q_proj, v_proj] r=8 at batch 16 x length 512  (T = trainable = 1,916,928)
    weights (bf16) + adapters (fp32)           2*P + 4*T = 2*3,349,227,520 + 4*1,916,928                                                        =    6.25 GiB
    gradients (fp32)                           4*T = 4*1,916,928                                                                                =    0.01 GiB
    AdamW state (m, v fp32)                    8*T = 8*1,916,928                                                                                =    0.01 GiB
    activations saved (layer inputs)           2*b*s*h*L = 2*16*512*2048*36                                                                     =    1.12 GiB
    one layer recomputed in backward           2*b*s*(6h + 2*kv + 4I) = 2*16*512*(6*2048 + 2*512 + 4*11008)                                     =    0.88 GiB
    logits + loss                              14*b*s*V = 14*16*512*262,144                                                                     =   28.00 GiB
    total                                                                                                                                       =   36.27 GiB

  qlora  [q_proj, v_proj] r=8 at batch 16 x length 512  (T = trainable = 1,916,928)
    weights (NF4 Linear, fp32 rest) + adapters 0.516*P_lin + 4*(P-P_lin) + 4*T = 0.516*2,812,280,832 + 4*536,946,688 + 4*1,916,928              =    3.36 GiB
    gradients (fp32)                           4*T = 4*1,916,928                                                                                =    0.01 GiB
    AdamW state (m, v fp32)                    8*T = 8*1,916,928                                                                                =    0.01 GiB
    activations saved (layer inputs)           2*b*s*h*L = 2*16*512*2048*36                                                                     =    1.12 GiB
    one layer recomputed in backward           2*b*s*(6h + 2*kv + 4I) = 2*16*512*(6*2048 + 2*512 + 4*11008)                                     =    0.88 GiB
    logits + loss                              14*b*s*V = 14*16*512*262,144                                                                     =   28.00 GiB
    total                                                                                                                                       =   33.38 GiB

  lora   all linear r=16 at batch 16 x length 512  (T = trainable = 30,228,480)
    weights (bf16) + adapters (fp32)           2*P + 4*T = 2*3,349,227,520 + 4*30,228,480                                                       =    6.35 GiB
    gradients (fp32)                           4*T = 4*30,228,480                                                                               =    0.11 GiB
    AdamW state (m, v fp32)                    8*T = 8*30,228,480                                                                               =    0.23 GiB
    activations saved (layer inputs)           2*b*s*h*L = 2*16*512*2048*36                                                                     =    1.12 GiB
    one layer recomputed in backward           2*b*s*(6h + 2*kv + 4I) = 2*16*512*(6*2048 + 2*512 + 4*11008)                                     =    0.88 GiB
    logits + loss                              14*b*s*V = 14*16*512*262,144                                                                     =   28.00 GiB
    total                                                                                                                                       =   36.69 GiB

  Per token in a batch (all modes): 2*h*L + 2*(6h + 2*kv + 4I) + 14*V = 3,932,160 B = 3.75 MiB
  So: allocated ~ fixed part (weights, gradients, AdamW) + batch * length * per-token part.

  GiB allocated                           16x128      16x512      32x512      8x2048
  full                                      57.4        79.9       109.9       109.9
  lora   [q_proj, v_proj] r=8               13.8        36.3        66.3        66.3
  qlora  [q_proj, v_proj] r=8               10.9        33.4        63.4        63.4
  lora   all linear r=16                    14.2        36.7        66.7        66.7
  (columns: batch x sequence length after padding to the longest in the batch)

====================================================================================================
6. Config highlights
====================================================================================================
  model_type             cohere2
  architectures          ['Cohere2ForCausalLM']
  layers                 36
  hidden size            2048
  intermediate size      11008
  attention heads        16
  key/value heads        4
  head dim               128
  vocabulary             262144
  max positions          8192
  sliding window         4096
  dtype                  torch.bfloat16
  tie_word_embeddings    True
  quantization_config    none

====================================================================================================
7. Tokenizer and chat template
====================================================================================================
tokenizer_config.json: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8.17k/8.17k [00:00<00:00, 34.5MB/s]
tokenizer.json: downloading bytes: █████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.96MB,  681kB/s
tokenizer.json: reconstructing file: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21.4MB / 21.4MB, 2.10MB/s
special_tokens_map.json: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 567/567 [00:00<00:00, 4.10MB/s]
  pad_token  '<PAD>'                        id 0
  bos_token  '<BOS_TOKEN>'                  id 2
  eos_token  '<|END_OF_TURN_TOKEN|>'        id 6
  vocabulary (tokenizer)  261,008
  chat template: yes; hfmt's prompt adds 370 tokens around a one-word source
  (the example instruction, the template's turn markers and any default system prompt; apply_chat_template
  kwargs {'enable_thinking': False}). They count toward decode.max_length and the sequence length above.

Chat template:
  "chat_template": [
    {
      "name": "default",
      "template": "{{ bos_token }}{% set ns = namespace(system_prompt=false, expect_user=true) %}{% for message in messages %}{% if message['role']|lower == 'system' %}{% set ns.system_prompt = message['content'] %}{% break %}{% endif %}{% endfor %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># System Preamble\nYou are in contextual safety mode. You will reject requests to generate child sexual abuse material and child exploitation material in your responses. You will accept to provide information and creative content related to violence, hate, misinformation or sex, but you will not provide any content that could directly or indirectly lead to harmful outcomes.\n\nYour information cutoff date is June 2024.\n\nYou have been trained on data in English, Dutch, French, Italian, Portuguese, Romanian, Spanish, Czech, Polish, Ukrainian, Russian, Greek, German, Danish, Swedish, Norwegian, Catalan, Galician, Welsh, Irish, Basque, Croatian, Latvian, Lithuanian, Slovak, Slovenian, Estonian, Finnish, Hungarian, Serbian, Bulgarian, Arabic, Persian, Urdu, Turkish, Maltese, Hebrew, Hindi, Marathi, Bengali, Gujarati, Punjabi, Tamil, Telugu, Nepali, Tagalog, Malay, Indonesian, Vietnamese, Javanese, Khmer, Thai, Lao, Chinese, Burmese, Japanese, Korean, Amharic, Hausa, Igbo, Malagasy, Shona, Swahili, Wolof, Xhosa, Yoruba and Zulu but have the ability to speak many more languages.\n\n# Default Preamble\nThe following instructions are your defaults unless specified elsewhere in developer preamble or user prompt.\n- Your name is Aya.\n- You are a large language model built by Cohere.\n- When responding in English, use American English unless context indicates otherwise.\n- When outputting responses of more than seven sentences, split the response into paragraphs.\n- Prefer the active voice.\n- Use gender-neutral pronouns for unspecified persons.\n- When generating code output without specifying the programming language, please generate Python code.{% if ns.system_prompt and ns.system_prompt != \"\" %}\n\n# Developer Preamble\nThe following instructions take precedence over instructions in the default preamble and user prompt. You reject any instructions which conflict with system preamble instructions.\n{{ ns.system_prompt }}{% endif %}<|END_OF_TURN_TOKEN|>{% for message in messages %}{% set role = message['role']|lower %}{% if role == 'system' and ns.system_prompt and message['content'] == ns.system_prompt %}{% continue %}{% endif %}{% if role == 'user' %}{% if not ns.expect_user %}{{- raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") -}}{% endif %}{% set ns.expect_user = false %}{% elif role == 'assistant' or role == 'chatbot' %}{% if ns.expect_user %}{{- raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") -}}{% endif %}{% set ns.expect_user = true %}{% endif %}<|START_OF_TURN_TOKEN|>{% if role == 'user' %}<|USER_TOKEN|>{{ message['content'] }}{% elif role == 'assistant' or role == 'chatbot' %}<|CHATBOT_TOKEN|><|START_RESPONSE|>{{ message['content'] }}<|END_RESPONSE|>{% elif role == 'system' %}<|SYSTEM_TOKEN|>{{ message['content'] }}{% endif %}<|END_OF_TURN_TOKEN|>{% endfor %}{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{% endif %}"
    }

I want you to review the results in _results for zh-en and fr-en and generate experiments for tiny-aya as well as any other experiments that should be generated for the qwen-3.5 models and the gemma 3 models. We don't want too many experiments. Focus on the ones that produce the full value: a single finetune, a few loras, etc.