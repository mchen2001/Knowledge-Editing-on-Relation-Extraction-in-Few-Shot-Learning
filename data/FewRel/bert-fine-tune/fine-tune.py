import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os

# os.chdir("FewRel/bert-fine-tune")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FewRelDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['tokens']
        label = item['relation']

        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoding = {key: val.squeeze() for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

with open('2/0.json', 'r') as f:
    train_data = json.load(f)

with open('test_bert.json', 'r') as f:
    test_data = json.load(f)

# Load tokenizer and add special tokens
special_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer.add_tokens(special_tokens)

# Compute the number of unique relation labels
NUM_LABELS = len(set(item['relation'] for item in train_data + test_data))
# print(NUM_LABELS)

# Load model and resize token embeddings
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

MAX_LENGTH = 512
train_dataset = FewRelDataset(train_data, tokenizer, max_length=MAX_LENGTH)
test_dataset = FewRelDataset(test_data, tokenizer, max_length=MAX_LENGTH)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=40,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    warmup_ratio=0.1,
    weight_decay=0.0,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    load_best_model_at_end=True,
    lr_scheduler_type='linear'
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Fine-tune model
trainer.train()

# Evaluate model
results = trainer.evaluate()

# Save model
trainer.save_model('./fine-tuned-bert')

print("Training complete!")
print("Evaluation results:", results)
