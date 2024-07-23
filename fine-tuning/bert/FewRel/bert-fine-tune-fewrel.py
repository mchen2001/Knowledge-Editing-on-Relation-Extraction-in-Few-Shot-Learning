import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, AdamW
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.metrics import accuracy_score
import random

os.chdir("./fine-tuning/")

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

def run_fine_tune(n, i):
    print(f"Start training on {n}/{i}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    with open(f'../data/FewRel/bert-fine-tune/{n}/{i}.json', 'r') as f:
        train_data = json.load(f)

    with open('../../../data/FewRel/bert-fine-tune/test_bert.json', 'r') as f:
        test_data = json.load(f)

    random.shuffle(train_data)
    random.shuffle(test_data)

    special_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_tokens(special_tokens)

    NUM_LABELS = len(set(item['relation'] for item in train_data + test_data))
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_LABELS)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    MAX_LENGTH = 512
    train_dataset = FewRelDataset(train_data, tokenizer, max_length=MAX_LENGTH)
    test_dataset = FewRelDataset(test_data, tokenizer, max_length=MAX_LENGTH)
    number_of_examples = len(train_dataset)
    per_device_train_batch_size = 32
    num_devices = 1
    steps_per_epoch = number_of_examples // (per_device_train_batch_size * num_devices)
    save_every_n_epochs = 5
    save_steps = steps_per_epoch * save_every_n_epochs
    print(save_steps)



    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {'accuracy': accuracy_score(p.label_ids, preds)}

    training_args = TrainingArguments(
        output_dir='results',
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
        load_best_model_at_end=True,
        lr_scheduler_type='linear'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=(AdamW(model.parameters(), lr=2e-5, weight_decay=0.01), None)
    )

    trainer.train()

    results = trainer.evaluate()
    trainer.save_model(f'./fine-tuned-bert-model/{n}/{i}')

    print("Training complete!")
    print("Evaluation results:", results)

def main():
    run_fine_tune(16,0)
    # for n in [2, 4, 8, 16, 32]:
    #     for i in range(10):
    #         run_fine_tune(n,i)


if __name__ == "__main__":
    main()