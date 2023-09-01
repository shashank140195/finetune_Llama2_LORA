# finetune_Llama2_LORA
This repository contains the work to finetune Llama-2 7B HF model using LoRA on 1 A100 40GB GPU. All the Experiments are done in Google Colab Pro+ using 1 A100 40GB  GPU.

# Requesting model access from META
### 1. Requesting model access from META
visit this [link](https://ai.meta.com/llama/) and request the access to the Llama-2 models.  

### 2. Requesting model access from Hugging Face
Once request is approved, use the same email address to get the access of the model from HF [here](https://huggingface.co/meta-llama/Llama-2-7b).  

Once both requests are approved, follow the below directions. 

# Dataset
The full modified raw dataset is available at this [link](https://drive.google.com/drive/folders/1XkfRKwWdrrV-wdzp9GdEXJHTHit9GbNi?usp=sharing). This dataset has been modified and made error free and has been released publically by me for further research. 

## 1. Preparing data for the model
We create the prompt for the model as:  
```
instruction:as a relation extractor assistant, identify the relations in this document. do not generate any tokens outside of the document.
document:lbsl is an extremely rare disorder that was first reported in the medical literature in 2002. according to the nonprofit organization, a cure for ellie, as of april 2018, there are about 100 individuals worldwide who have been identified with the disorder. because rare diseases like lbsl often go undiagnosed or misdiagnosed, it is difficult to determine the true frequency in the general population.
output: thank you for the information.the relationship between raredisease lbsl and anaphor "the disorder" is antecedent.
```  
To create the data like above, download the dataset from above link, run the sctipts from [Databuilder](https://github.com/shashank140195/finetune_Llama2_LORA/tree/main/dataBuilder) in the order   
1. jsonbuilder.py to get json files files  
2. csvbuilder.py which takes json file as input and get the .csv files for the model.  

Alternatively you can use the already created .csv files from [datsetfolder](https://github.com/shashank140195/finetune_Llama2_LORA/tree/main/dataset)

## 2. Environment preparation
```
!git clone https://github.com/facebookresearch/llama-recipes.git

import os
os.chdir('/content/llama-recipes')
!pip install git+https://github.com/huggingface/transformers

# python 3.9 or higher recommended
!pip install -r requirements.txt
```  

## 3. Authorising HF token
Once HF request to access the model has been approved, create hugging face token [here](https://huggingface.co/settings/tokens)

Run below cell and enter your token. It will authenticate your HF account
```
!huggingface-cli login
```

## 4. Download the model
```
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model =LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=True, device_map='auto', torch_dtype=torch.float16)
```

## 5. Creating LoRA config
```
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
        r=3000,
        lora_alpha=16,
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
```
## 6. Setting up profiler
```
from transformers import TrainerCallback
from contextlib import nullcontext
enable_profiler = False
output_dir = "/content/llama-output"

config = {
    'lora_config': lora_config,
    'learning_rate': 2e-4,
    'num_train_epochs': 10,
    'gradient_accumulation_steps': 4,
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
```

## 7. Preparing dataset for Llama-2 input
```
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

train_dataset = get_dataset("/content/train.csv")
validate_dataset = get_dataset("/content/valid.csv")
```

## 8. Start training (fine-tuning) the model
```
from transformers import default_data_collator, Trainer, TrainingArguments

# Define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    bf16=True,  # Use BF16 if available
    # logging strategies
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=50,
    save_strategy= "steps",
    per_device_eval_batch_size= 2,
    optim="adamw_torch_fused",
    eval_steps=50,
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
```

Alternatively you can find our working script [here](https://github.com/shashank140195/finetune_Llama2_LORA/blob/main/scripts/finetune.py)

## 9. Inference
Once model is trained, run the [inference](https://github.com/shashank140195/finetune_Llama2_LORA/blob/main/scripts/llama2_inference.py) or below code snippet to make the predictions.
```
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

PATH_TO_YOUR_TRAINED_MODEL_CHECKPOINT = "checkpoint"
model = AutoModelForCausalLM.from_pretrained(PATH_TO_YOUR_TRAINED_MODEL_CHECKPOINT)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

df = pd.read_csv("/content/test.csv")
df['prediction'] = None

model.eval()
with torch.no_grad():

  for i in range(len(df)):
    text = df.iloc[i]["text"]
    abc = text.split("output:")
    input_prompt = abc[0] +"output:"
    model_input = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    prediction = tokenizer.decode(model.generate(**model_input, max_new_tokens=1024)[0], skip_special_tokens=True)
    df["prediction"][i] = prediction

#path to save the predictions
df.to_csv("prediction.csv", index=False)

```

## 10. Evaluation
Once you save the predictions.csv file, run [eval](https://github.com/shashank140195/finetune_Llama2_LORA/blob/main/scripts/eval.py) to get the scores on the dataset. 

We reached the score of 31 F1 with 1500 steps of training using lr 1e-4, rank = 2500, and lora_alpha = 32, beating BioGPT for RE task. 
