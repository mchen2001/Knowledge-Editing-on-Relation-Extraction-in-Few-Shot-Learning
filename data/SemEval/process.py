import pandas as pd
import numpy as np
import os

os.chdir("./data/SemEval/")

def load_and_simplify_semeval2010_task8(file_path):
    sentences = []
    relations = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):
            sentence = lines[i].strip().split('\t')[1].strip('\"')
            relation = lines[i+1].strip().split('(')[0]
            sentence = normalize_sentence(sentence)
            sentences.append(f"[CLS] {sentence} [SEP]")
            relations.append(relation)
    
    return pd.DataFrame({'sentence': sentences, 'relation': relations})

def normalize_sentence(sentence):
    sentence = sentence.replace('<e1>', '<SUBJ_START> ').replace('</e1>', ' <SUBJ_END>')
    sentence = sentence.replace('<e2>', '<OBJ_START> ').replace('</e2>', ' <OBJ_END>')
    return sentence

train_data = load_and_simplify_semeval2010_task8('TRAIN_FILE.TXT')
test_data = load_and_simplify_semeval2010_task8('TEST_FILE_FULL.TXT')

num_shots_list = [2, 4, 8, 16, 32, 64, 128, 256, 512]
seed = 0

def sample_few_shot_data(train_data, num_shots, seed):
    np.random.seed(seed)
    sampled_data = train_data.groupby(
        'relation', group_keys=False
        ).apply(
            lambda x: x.sample(min(num_shots, len(x)), replace=False)
            ).reset_index(drop=True)
    return sampled_data

for num_shots in num_shots_list:
    sampled_data = sample_few_shot_data(train_data, num_shots, seed)
    output_file = f'few_shot_samples/{num_shots}.json'
    sampled_data.to_json(output_file, orient='records', lines=True)
    print(f"Few-shot samples with {num_shots} examples saved to {output_file}")

test_data.to_json("test_data.json", orient='records', lines=True)