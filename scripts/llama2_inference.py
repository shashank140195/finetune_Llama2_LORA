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

df.to_csv("prediction.csv", index=False)