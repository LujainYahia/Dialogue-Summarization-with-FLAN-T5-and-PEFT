# Dialogue-Summarization-with-FLAN-T5-and-PEFT

This repository contains a script for training and evaluating a dialogue summarization model using the FLAN-T5 architecture with Parameter Efficient Fine-Tuning (PEFT). The script utilizes the `samsum` dataset and demonstrates both zero-shot and fine-tuned summarization capabilities.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)


## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/dialogue-summarization.git
   cd dialogue-summarization
   ```

2. **Install the Dependencies**

   ```bash
   pip install datasets transformers evaluate rouge_score py7zr accelerate torch peft
   ```

## Dataset Preparation

1. **Load the Dataset**

   The `samsum` dataset is loaded using the Hugging Face `datasets` library.

   ```python
   from datasets import load_dataset

   huggingface_dataset_name = "samsum"
   dataset = load_dataset(huggingface_dataset_name)
   ```

2. **Explore the Dataset**

   ```python
   index = 200
   dialogue = dataset['test'][index]['dialogue']
   summary = dataset['test'][index]['summary']

   print(f'Dialogue: {dialogue}')
   print(f'Summary: {summary}')
   ```

## Model Architecture

The model architecture is based on the FLAN-T5 model for sequence-to-sequence tasks.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name='google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Training

1. **Tokenize the Data**

   ```python
   def tokenize_function(example):
       start_prompt = 'Summarize the following conversation.\n\n'
       end_prompt = '\n\nSummary: '
       prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
       example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
       example['labels'] = tokenizer(example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
       return example

   tokenized_datasets = dataset.map(tokenize_function, batched=True)
   tokenized_datasets = tokenized_datasets.remove_columns(['id',  'dialogue', 'summary'])
   ```

2. **Training Arguments and Trainer**

   ```python
   from transformers import TrainingArguments, Trainer

   output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

   training_args = TrainingArguments(
       output_dir=output_dir,
       auto_find_batch_size=True,
       learning_rate=5e-05,
       num_train_epochs=15,
       per_device_train_batch_size=4,
       per_device_eval_batch_size=2,
       evaluation_strategy="epoch",
       logging_steps=2,
       max_steps=10
   )

   trainer = Trainer(
       model=original_model,
       args=training_args,
       train_dataset=tokenized_datasets['train'],
       eval_dataset=tokenized_datasets['validation']
   )

   trainer.train()
   trainer.save_model("./fine_tuned.pth")
   ```

## Parameter Efficient Fine-Tuning (PEFT)

1. **Configure and Apply PEFT**

   ```python
   from peft import LoraConfig, get_peft_model, TaskType

   lora_config = LoraConfig(
       r=32,
       lora_alpha=32,
       target_modules=["q", "v"],
       lora_dropout=0.1,
       bias="lora_only",
       task_type=TaskType.SEQ_2_SEQ_LM
   )

   peft_model = get_peft_model(original_model, lora_config)
   ```

2. **Training with PEFT**

   ```python
   peft_training_args = TrainingArguments(
       output_dir=f'./peft-dialogue-summary-training-{str(int(time.time()))}',
       auto_find_batch_size=True,
       learning_rate=5e-05,
       num_train_epochs=15,
       per_device_train_batch_size=4,
       per_device_eval_batch_size=2,
       evaluation_strategy="epoch",
       logging_steps=2,
       max_steps=10
   )

   peft_trainer = Trainer(
       model=peft_model,
       args=peft_training_args,
       train_dataset=tokenized_datasets["train"],
       eval_dataset=tokenized_datasets["validation"]
   )

   peft_trainer.train()
   peft_trainer.model.save_pretrained("./peft-dialogue-summary-checkpoint-local")
   tokenizer.save_pretrained("./peft-dialogue-summary-checkpoint-local")
   ```

## Evaluation

1. **Zero-shot Evaluation**

   ```python
   inputs = tokenizer(prompt, return_tensors='pt')
   output = tokenizer.decode(
       original_model.generate(
           inputs["input_ids"],
           max_new_tokens=200,
       )[0],
       skip_special_tokens=True
   )
   print(f'Zero-shot Output: {output}')
   ```

2. **PEFT Evaluation**

   ```python
   input_ids = tokenizer(prompt, return_tensors="pt").input_ids
   peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
   peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

   print(f'PEFT Output: {peft_model_text_output}')
   ```

## Usage

1. **Print Number of Trainable Parameters**

   ```python
   def print_number_of_trainable_model_parameters(model):
       trainable_model_params = 0
       all_model_params = 0
       for _, param in model.named_parameters():
           all_model_params += param.numel()
           if param.requires_grad:
               trainable_model_params += param.numel()
       return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

   print(print_number_of_trainable_model_parameters(original_model))
   print(print_number_of_trainable_model_parameters(peft_model))
   ```

---

This README provides a comprehensive guide to setting up, training, and evaluating a FLAN-T5-based dialogue summarization model with PEFT. For any further queries, please refer to the official [Hugging Face Transformers documentation](https://huggingface.co/transformers/).
