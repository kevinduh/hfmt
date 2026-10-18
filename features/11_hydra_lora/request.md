I want to clean up the current method for setting lora target_modules. A string is passed via hydra and the some logic inside sft_translation.py picks the right target module (line "    if finetune_mode in ("lora", "qlora"):"). This is very brittle. For a different base model, there may be different modules. So I want to just simplify things: first, create a hydra feature for target_modules that is an optional list of strings. lora_default.yaml will then set this like:

```
lora:
- target_modules
  - q_proj
  - v_proj
```

We will get rid of the "all-linear" option and just assume the user will always list out each module.
No need to be backward compatible. 

Additional request:
- While we're I'd like to rename all lora field names: Currently we have peft.lora.lora_r - it should just be pref.lora.r for simplicity
- To support knowing which modules a basemodel checkpoint has, please write a simple inspect_model.py script in tools/ that (1) prints the entire model, (2) lists all the module types and their frequencies, as well as total parameter count. (3) any other interesting statistic you think would be useful to report. This should just be a simple script I can call like: python inspect_model.py qwen3-1.5b
