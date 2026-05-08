#!/usr/bin/env python
import sys
import transformers, torch, datasets, accelerate, trl, bitsandbytes, peft

print(f"Python version: {sys.version}")
print(f"{transformers.__version__ =} tested: 5.8.0")
print(f"{torch.__version__ =} tested: 2.11.0+cu130")
print(f"{datasets.__version__ =} tested: 4.8.5")
print(f"{accelerate.__version__ =} tested: 1.13.0")
print(f"{trl.__version__ =} tested: 1.3.0")
print(f"{bitsandbytes.__version__ =} tested: 0.49.2")
print(f"{peft.__version__ =} tested: 0.19.1")
