import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.metrics import accuracy_score
import random
from collections import Counter
import math

os.chdir("./fine-tuning/bert/SemEval/")

class SemEvalDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, label_to_id):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['sentence']
        label = self.label_to_id[item['relation']]

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
    
def load_json_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def run_fine_tune(n):
    print(f"Start training on {n}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_data = load_json_file(f'../../../data/SemEval/few_shot_samples/{n}.json')
    test_data = load_json_file('../../../data/SemEval/test_data.json')

    random.shuffle(train_data)
    random.shuffle(test_data)

    special_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(special_tokens)

    with open('../../../data/SemEval/rel_to_id.json', 'r') as f:
        label_to_id = json.load(f)

    NUM_LABELS = len(label_to_id)
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    MAX_LENGTH = 512
    train_dataset = SemEvalDataset(train_data, tokenizer, max_length=MAX_LENGTH, label_to_id=label_to_id)
    test_dataset = SemEvalDataset(test_data, tokenizer, max_length=MAX_LENGTH, label_to_id=label_to_id)
    number_of_examples = len(train_dataset)
    per_device_train_batch_size = 32
    num_devices = 1
    steps_per_epoch = math.ceil(number_of_examples / (per_device_train_batch_size * num_devices))
    save_every_n_epochs = 5
    save_steps = steps_per_epoch * save_every_n_epochs
    print(save_steps)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids

        label_counts = Counter(labels)
        total_count = sum(label_counts.values())

        sample_weights = np.array([total_count / label_counts[label] for label in labels])

        return {'accuracy': accuracy_score(labels, preds, sample_weight=sample_weights)}


    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=20,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_steps=save_steps,
        lr_scheduler_type='linear',
        logging_first_step=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01), None)
    )

    trainer.train()

    results = trainer.evaluate()
    trainer.save_model(f'./fine-tuned-bert-model/{n}')

    print("Training complete!")
    print("Evaluation results:", results)

def main():
    for n in [2, 4, 8, 16, 32, 64, 128, 256]:
        run_fine_tune(n)

if __name__ == "__main__":
    main()