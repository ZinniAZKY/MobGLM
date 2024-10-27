import torch
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments


class CustomTextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=256):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.examples = []
        for line in lines:
            line = line.strip()
            if line:
                tokens = tokenizer(
                    line,
                    max_length=block_size,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                self.examples.append({
                    'input_ids': tokens.input_ids.squeeze(),
                    'attention_mask': tokens.attention_mask.squeeze(),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# Load the tokenizer and model
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Add special tokens if needed
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Prepare the dataset
dataset = CustomTextDataset(
    tokenizer,
    "path_to_your_dataset.txt",  # Update with your dataset path
    block_size=256  # Adjust block size as needed
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./gpt2-finetuned',
    overwrite_output_dir=True,
    num_train_epochs=5,  # Adjust the number of epochs
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,  # Simulate larger batch size
    save_steps=1_000,
    save_total_limit=2,
    logging_steps=500,
    logging_dir='./logs',
    report_to="none",
    fp16=True,  # Enable if your GPU supports it
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model('./gpt2-finetuned')
tokenizer.save_pretrained('./gpt2-finetuned')
