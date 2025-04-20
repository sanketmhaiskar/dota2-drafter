---
library_name: peft
license: mit
base_model: microsoft/phi-3-mini-4k-instruct
tags:
- generated_from_trainer
datasets:
- /home/sanket/dota2trainingData/train_alpaca.jsonl
model-index:
- name: outputs/phi3-draft-model
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
<details><summary>See axolotl config</summary>

axolotl version: `0.8.0`
```yaml
base_model: microsoft/phi-3-mini-4k-instruct
model_type: phi3
tokenizer_type: AutoTokenizer

load_in_8bit: true  # Use 8-bit if needed, or set to false if you're using LoRA with full precision
trust_remote_code: true

datasets:
  - path: /home/sanket/dota2trainingData/train_alpaca.jsonl
    type: alpaca
    validation_path: /home/sanket/dota2trainingData/test_alpaca.jsonl

dataset_prepared_path: ./data/phi3-draft-prepared
val_set_size: 0

output_dir: ./../outputs/phi3-draft-model

sequence_len: 512
sample_packing: false
pad_to_sequence_len: false

adapter: lora  # Use 'lora' instead of 'qlora'

lora_r: 8                # LoRA rank
lora_alpha: 16           # Scaling factor
lora_dropout: 0.05       # Dropout
lora_target_modules:
lora_target_modules:
  - self_attn.o_proj
  - self_attn.qkv_proj

micro_batch_size: 2
gradient_accumulation_steps: 2
num_epochs: 1
optimizer: adamw_torch
lr_scheduler: cosine
learning_rate: 1e-4

train_on_inputs: true
group_by_length: false

bf16: true
fp16: false  # Use fp16 for mixed precision training

logging_steps: 100
save_steps: 1000
evals_per_epoch: 1
save_total_limit: 1

prompt_template: alpaca

```

</details><br>

# outputs/phi3-draft-model

This model is a fine-tuned version of [microsoft/phi-3-mini-4k-instruct](https://huggingface.co/microsoft/phi-3-mini-4k-instruct) on the /home/sanket/dota2trainingData/train_alpaca.jsonl dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 4
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 100
- num_epochs: 1.0

### Training results



### Framework versions

- PEFT 0.15.1
- Transformers 4.51.3
- Pytorch 2.5.1
- Datasets 3.5.0
- Tokenizers 0.21.1