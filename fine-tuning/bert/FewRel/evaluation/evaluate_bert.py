import os
import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


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

test_path = '../../../../data/FewRel/bert-fine-tune/test_bert.json'
with open(test_path, 'r') as file:
    test_data = json.load(file)

MAX_LEN = 512

def evaluate_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).flatten()
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')

    return acc, f1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results = []

for n in [2,4,8,16,32,64,128,256]:
    model_name = f"{n}"
    tokenizer_path = f'../fine-tuned-bert-model/{n}'
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    test_dataset = FewRelDataset(test_data, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model_path = tokenizer_path
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    acc, f1 = evaluate_model(model, test_loader)
    results.append((model_name, acc, f1))
    print(f'Model: {model_name} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}')


df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score'])
df.to_csv('evaluation_results.csv', index=False)
print("Evaluation results saved to evaluation_results.csv")