import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# model_id="meta-llama/Llama-2-7b-chat-hf"
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
# model =LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
model =LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto')


model.train()

def create_peft_config(model):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_int8_training,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=2700,
        lora_alpha=8,
        lora_dropout=0,
        target_modules = ["q_proj", "v_proj"]
    )

    # prepare int-8 model for training
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, peft_config

# create peft config
model, lora_config = create_peft_config(model)

from transformers import TrainerCallback
from contextlib import nullcontext
enable_profiler = False
output_dir = "/home/ubuntu/llama-recipes/llama_output"

config = {
    'lora_config': lora_config,
    'learning_rate': 1e-4,
    'num_train_epochs': 5,
    'gradient_accumulation_steps': 16,
    'per_device_train_batch_size': 1,
    'gradient_checkpointing': False,
}

# Set up profiler
if enable_profiler:
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"{output_dir}/logs/tensorboard"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True)

    class ProfilerCallback(TrainerCallback):
        def __init__(self, profiler):
            self.profiler = profiler

        def on_step_end(self, *args, **kwargs):
            self.profiler.step()

    profiler_callback = ProfilerCallback(profiler)
else:
    profiler = nullcontext()


from transformers import AutoTokenizer
from datasets import Dataset
import pandas as pd

def get_dataset(file_path):
  # 1. Load the CSV file
  df = pd.read_csv(file_path)

  # 2. Initialize the tokenizer and tokenize the "text" column
  # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
  tokenizer.pad_token = tokenizer.eos_token
  tokenized_data = tokenizer(df['text'].tolist(), truncation=True, padding=True, return_tensors="pt")

  # 3. Convert the tokenized data into the HuggingFace dataset format
  dataset_dict = {
      "input_ids": tokenized_data["input_ids"].numpy(),
      "attention_mask": tokenized_data["attention_mask"].numpy(),
      "labels": tokenized_data["input_ids"].numpy()  # copying input_ids to labels
  }

  # 4. Convert the dictionary into a HuggingFace Dataset
  dataset = Dataset.from_dict(dataset_dict)
  return dataset

train_dataset = get_dataset("/home/ubuntu/llama-recipes/train.csv")
validate_dataset = get_dataset("/home/ubuntu/llama-recipes/valid.csv")

from transformers import default_data_collator, Trainer, TrainingArguments

# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    # bf16=True,  # Use BF16 if available
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy= "steps",
    per_device_eval_batch_size= 2,
    optim="adamw_torch_fused",
    eval_steps=10,
    save_total_limit=4,
    save_steps= 50,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    do_eval=True,
    evaluation_strategy="steps",
    max_steps=total_steps if enable_profiler else -1,
    **{k:v for k,v in config.items() if k != 'lora_config'}
)

with profiler:
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validate_dataset,
        data_collator=default_data_collator,
        callbacks=[profiler_callback] if enable_profiler else [],
    )

    # Start training
    trainer.train()