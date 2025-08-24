import os
# Use a writable temporary cache folder
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_HOME"] = "/tmp/huggingface_cache"  # for HF token/config
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface_cache"
os.environ["HF_METRICS_CACHE"] = "/tmp/huggingface_cache"

import streamlit as st
from datasets import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import math
from peft import LoraConfig, get_peft_model

os.environ["WANDB_MODE"] = "disabled"

# -----------------------------
# SETTINGS
# -----------------------------

st.set_page_config(page_title="My Fine-Tuned Model", layout="centered")

st.title("Chat with My Fine-Tuned Model 🤖")

# Hugging Face model ID
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL_ID)

# For private models: set HF_TOKEN as environment variable or secret
HF_TOKEN = os.getenv("HF_TOKEN")  # leave None if public

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD MODEL & TOKENIZER
# -----------------------------

class LoraLinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.05, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 1.0

        # Base frozen linear layer
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_A, self.lora_B, self.lora_dropout = None, None, None

    def forward(self, x):
        # Base forward
        result = F.linear(x, self.weight, self.bias)

        # LoRA adaptation
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + self.scaling * lora_out

        return result

class MoELoRALinear(nn.Module):
    def __init__(self, base_linear, r, num_experts=2, k=1, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.base_linear = base_linear  # <-- frozen pretrained weight
        self.num_experts = num_experts
        self.k = k

        self.experts = nn.ModuleList([
            LoraLinear(   # LoRA adapter only
                in_features=base_linear.in_features,
                out_features=base_linear.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            for _ in range(num_experts)
        ])

        self.gate = nn.Linear(base_linear.in_features, num_experts)

    def forward(self, x):
        # keep frozen pretrained path
        base_out = self.base_linear(x)

        # gating for experts
        gate_scores = torch.softmax(self.gate(x), dim=-1)

        expert_out = 0
        for i, expert in enumerate(self.experts):
            expert_out += gate_scores[..., i:i+1] * expert(x)

        return base_out + expert_out

def replace_proj_with_moe_lora(model, r=8, num_experts=2, k=1, lora_alpha=16, lora_dropout=0.05):
    """
    Replace only up_proj, down_proj, in each MLP with MoE(LoRA) versions.
    """
    for layer in model.model.layers:
        for proj_name in ["up_proj", "down_proj"]:
            old = getattr(layer.mlp, proj_name)
            moe = MoELoRALinear(
                base_linear=old,
                r=r,
                num_experts=num_experts,
                k=k,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            ).to(next(old.parameters()).device)
            setattr(layer.mlp, proj_name, moe)

    return model

def preprocess(example):
    # Tokenize full text
    tokens = tokenizer(example['text'], truncation=True, padding=False)

    # Find indices to mask
    # Everything before <|assistant|> is ignored in loss
    text = example['text']
    assistant_index = text.find("<|assistant|>")

    # Convert character index to token index
    prefix_ids = tokenizer(text[:assistant_index], add_special_tokens=False)['input_ids']
    prefix_len = len(prefix_ids)

    # Prepare labels: -100 for question/system tokens
    labels = tokens['input_ids'].copy()
    labels[:prefix_len] = [-100] * prefix_len

    tokens['labels'] = labels
    return tokens

def load_model(model_id, token=None):
    file_path = 'makemytrip_qa_full.json'
    try:
        df = pd.read_json(file_path)
        display(df.head())
    except FileNotFoundError:
        st.write(f"Error: File not found at {file_path}. Please check the path.")
        data = [
                  {
                    "question": "What were MakeMyTrip's total assets as of March 31, 2024?",
                    "answer": "MakeMyTrip's total assets as of March 31, 2024 were USD 1,660,077 thousand."
                  },
                  {
                    "question": "How much did MakeMyTrip report as total assets at the end of March 2024?",
                    "answer": "MakeMyTrip's total assets as of March 31, 2024 were USD 1,660,077 thousand."
                  },
                  {
                    "question": "What is the figure for total assets on March 31, 2024?",
                    "answer": "MakeMyTrip's total assets as of March 31, 2024 were USD 1,660,077 thousand."
                  },
                  {
                    "question": "What were MakeMyTrip's total assets as of March 31, 2025?",
                    "answer": "MakeMyTrip's total assets as of March 31, 2025 were USD 1,828,288 thousand."
                  },
                  {
                    "question": "Can you tell me the total assets reported by MakeMyTrip at the end of March 2025?",
                    "answer": "MakeMyTrip's total assets as of March 31, 2025 were USD 1,828,288 thousand."
                  },
                  {
                    "question": "How much did MakeMyTrip list as total assets on March 31, 2025?",
                    "answer": "MakeMyTrip's total assets as of March 31, 2025 were USD 1,828,288 thousand."
                  },
                  {
                    "question": "What was MakeMyTrip's total revenue for the year ended March 31, 2024?",
                    "answer": "MakeMyTrip's total revenue for the year ended March 31, 2024 was USD 782,524 thousand."
                  },
                  {
                    "question": "How much revenue did MakeMyTrip generate in the fiscal year 2024?",
                    "answer": "MakeMyTrip's total revenue for the year ended March 31, 2024 was USD 782,524 thousand."
                  },
                  {
                    "question": "What is the reported revenue of MakeMyTrip for the year ending March 31, 2024?",
                    "answer": "MakeMyTrip's total revenue for the year ended March 31, 2024 was USD 782,524 thousand."
                  },
                  {
                    "question": "What was MakeMyTrip's total revenue for the year ended March 31, 2025?",
                    "answer": "MakeMyTrip's total revenue for the year ended March 31, 2025 was USD 978,336 thousand."
                  },
                  {
                    "question": "How much revenue did MakeMyTrip generate in the 2025 fiscal year?",
                    "answer": "MakeMyTrip's total revenue for the year ended March 31, 2025 was USD 978,336 thousand."
                  }
                ]
        df = pd.DataFrame(data)
    except Exception as e:
        st.write(f"An error occurred: {e}")
    
    # Prepare data for fine-tuning
    training_data = []
    
    # Define a system prompt for your domain
    system_prompt = "You are a helpful assistant that provides financial data from MakeMyTrip reports."
    
    # Correctly format each training example with the chat template
    for index, row in df.iterrows():
        question = row['question']
        answer = row['answer']
    
        # Format the data using the TinyLlama chat template
        training_data.append({
            "text": f"<|system|>\n{system_prompt}</s>\n<|user|>\n{question}</s>\n<|assistant|>\n{answer}</s>"
        })

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    st.write(transformers.__version__)
    st.write([name for name in dir(transformers) if "Tokenizer" in name])
    st.write(transformers.__file__)
    
    dataset = Dataset.from_list(training_data)
    tokenized_dataset = dataset.map(preprocess, remove_columns=["text"])

    model = replace_proj_with_moe_lora(
        base_model,
        r=8,
        num_experts=2,
        k=1,
        lora_alpha=16,
        lora_dropout=0.05
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    st.write(f"Trainable params: {trainable:,d} || Total params: {total:,d} || "
              f"Trainable%: {100 * trainable / total:.4f}")
    
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    training_args = TrainingArguments(
        learning_rate=5e-5,
        output_dir="./results",
        num_train_epochs=2,
        per_device_train_batch_size=1, # Keep batch size small
        gradient_accumulation_steps=4, # Increased gradient accumulation steps
        logging_steps=1,
        save_steps=10,
        save_total_limit=2,
        fp16=False, # fp16 and bf16 are mutually exclusive. bf16 is recommended for Ampere+ GPUs.
        bf16=True,  # Use bf16 for better performance with 4-bit models
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    st.write("Training started")
    trainer.train()
    
    model.eval()  # switch to evaluation mode
    return model

model = load_model(MODEL_ID, HF_TOKEN)

# -----------------------------
# USER INPUT
# -----------------------------
prompt = st.text_area("Enter your question:", height=150)
system_prompt = "You are a helpful assistant that provides financial data from MakeMyTrip reports."

max_tokens = st.slider("Max tokens to generate:", min_value=50, max_value=500, value=200, step=10)

if st.button("Generate Answer"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt!")
    else:
        # Tokenize input
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    
        # Apply the chat template to format the input
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True # This adds the <|assistant|> token at the end
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
        
        # Decode and display
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.success(answer)
