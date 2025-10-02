# train_bot.py

# 1️⃣ Import required libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# 2️⃣ Load dataset (first 1000 rows for testing)
dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k", split="train")
dataset = dataset.select(range(1000))  # Use only first 1000 rows for faster testing
print("Sample row:", dataset[0])

# 3️⃣ Preprocess dataset into dialogue format
def format_dialogue(example):
    # Combine customer input and agent response into a single text string
    return {"input_text": f"User: {example['input']} Bot: {example['output']}"}

dataset = dataset.map(format_dialogue)

# 4️⃣ Load pretrained DialoGPT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# ⚠ Fix: DialoGPT tokenizer has no pad_token by default
# Set pad_token to eos_token to enable padding
tokenizer.pad_token = tokenizer.eos_token

# 5️⃣ Tokenize the dataset and add labels
def tokenize(example):
    tokens = tokenizer(
        example["input_text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    # For causal LM, labels are the same as input_ids
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize, batched=True)

# Set the format for PyTorch training
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 6️⃣ Load pretrained DialoGPT model
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# 7️⃣ Define training arguments
training_args = TrainingArguments(
    output_dir="./helpdesk-bot",      # Directory to save model checkpoints
    per_device_train_batch_size=4,     # Adjust batch size based on GPU memory
    gradient_accumulation_steps=2,     # Helps if GPU memory is small
    num_train_epochs=1,                # Number of training epochs (increase for better results)
    logging_steps=50,                  # Log training progress every N steps
    save_steps=200,                    # Save checkpoint every N steps
    save_total_limit=2,                # Keep only last 2 checkpoints
    fp16=True,                         # Use mixed precision if GPU is available
    remove_unused_columns=False        # Keep all columns for causal LM
)

# 8️⃣ Create Trainer and start fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

# 9️⃣ Save fine-tuned model and tokenizer
model.save_pretrained("./helpdesk-bot")
tokenizer.save_pretrained("./helpdesk-bot")
print("Model saved in ./helpdesk-bot")


# code for google colab

# train_bot.py

# 1️⃣ Import required libraries
# import torch
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# # Check GPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"✅ Device available: {device}")
# if device == "cuda":
#     print(torch.cuda.get_device_name(0))
#     print(f"GPU Count: {torch.cuda.device_count()}")

# # 2️⃣ Load dataset (first 1000 rows for testing)
# dataset = load_dataset("MohammadOthman/mo-customer-support-tweets-945k", split="train")
# dataset = dataset.select(range(1000))  # Use only first 1000 rows for faster testing
# print("Sample row:", dataset[0])

# # 3️⃣ Preprocess dataset into dialogue format
# def format_dialogue(example):
#     return {"input_text": f"User: {example['input']} Bot: {example['output']}"}

# dataset = dataset.map(format_dialogue)

# # 4️⃣ Load pretrained DialoGPT tokenizer
# tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# # ⚠ Fix: DialoGPT tokenizer has no pad_token by default
# tokenizer.pad_token = tokenizer.eos_token

# # 5️⃣ Tokenize the dataset and add labels
# def tokenize(example):
#     tokens = tokenizer(
#         example["input_text"],
#         truncation=True,
#         padding="max_length",
#         max_length=128
#     )
#     tokens["labels"] = tokens["input_ids"].copy()
#     return tokens

# tokenized_dataset = dataset.map(tokenize, batched=True)
# tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# # 6️⃣ Load pretrained DialoGPT model
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium").to(device)

# # 7️⃣ Define training arguments
# training_args = TrainingArguments(
#     output_dir="./helpdesk-bot",      
#     per_device_train_batch_size=4,     
#     gradient_accumulation_steps=2,     
#     num_train_epochs=1,                
#     logging_steps=50,                  
#     save_steps=200,                    
#     save_total_limit=2,                
#     fp16=torch.cuda.is_available(),    # Use mixed precision if GPU available
#     remove_unused_columns=False
# )

# # 8️⃣ Create Trainer and start fine-tuning
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset
# )

# print("🚀 Starting fine-tuning...")
# trainer.train()
# print("✅ Fine-tuning complete!")

# # 9️⃣ Save fine-tuned model and tokenizer locally
# model.save_pretrained("./helpdesk-bot")
# tokenizer.save_pretrained("./helpdesk-bot")
# print("💾 Model saved locally at ./helpdesk-bot")

# # 🔟 Save model to Google Drive
# from google.colab import drive
# import shutil

# drive.mount('/content/drive')

# drive_path = "/content/drive/MyDrive/helpdesk-bot"
# shutil.copytree("./helpdesk-bot", drive_path, dirs_exist_ok=True)
# print(f"☁️ Model copied to Google Drive at: {drive_path}")
