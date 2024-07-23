import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import random
import json
import os

os.chdir('./FewRel/')



dataset = load_dataset('few_rel')
train_data = dataset['train_wiki']
val_data = dataset['val_wiki']

def convert_to_dataframe(data):
    records = []
    for item in data:
        relation = item['relation']
        tokens = item['tokens']
        head = item['head']['text']
        tail = item['tail']['text']
        records.append({
            'relation': relation,
            'tokens': tokens,
            'head': head,
            'tail': tail
        })
    return pd.DataFrame(records)

train_df = convert_to_dataframe(train_data)
val_df = convert_to_dataframe(val_data)

df = pd.concat([train_df, val_df]).reset_index(drop=True)

def split_data(df, test_size=0.1, random_state=42):
    train_data = []
    test_data = []
    labels = df['relation'].unique()
    for label in labels:
        label_data = df[df['relation'] == label]
        train, test = train_test_split(label_data, test_size=test_size, random_state=random_state)
        train_data.append(train)
        test_data.append(test)
    train_df = pd.concat(train_data).reset_index(drop=True)
    test_df = pd.concat(test_data).reset_index(drop=True)
    return train_df, test_df

def sample_data(df, n, seeds):
    sampled_data = {seed: [] for seed in seeds}
    labels = df['relation'].unique()
    for seed in seeds:
        random.seed(seed)
        for label in labels:
            label_data = df[df['relation'] == label]
            if len(label_data) >= n:
                sampled_data[seed].extend(label_data.sample(n, random_state=seed).to_dict('records'))
            else:
                sampled_data[seed].extend(label_data.to_dict('records'))
    return sampled_data

train_df, test_df = split_data(df)

sample_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
seeds = [i for i in range(10)]

sampled_data = {n: sample_data(train_df, n, seeds) for n in sample_sizes}

train_df.to_json('train_data.json', orient='records', lines=True)
test_df.to_json('test_data.json', orient='records', lines=True)
for n, data in sampled_data.items():
    for seed, samples in data.items():
        with open(f'few-shot-sample/{n}/{seed}.json', 'w') as file:
            json.dump(samples, file)

print("Data splitting and sampling completed successfully.")
