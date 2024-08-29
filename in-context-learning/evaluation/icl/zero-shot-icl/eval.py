import json
import pandas as pd
import re
import csv

FEWREL_IDX = {0: 'residence', 1: 'publisher', 2: 'participating team', 3: 'located in or next to body of water', 4: 'main subject', 5: 'instrument', 6: 'record label', 7: 'country of citizenship', 8: 'developer', 9: 'genre', 10: 'part of', 11: 'country of origin', 12: 'operating system', 13: 'language of work or name', 14: 'followed by', 15: 'heritage designation', 16: 'headquarters location', 17: 'occupant', 18: 'original language of film or TV show', 19: 'characters', 20: 'taxon rank', 21: 'work location', 22: 'mouth of the watercourse', 23: 'position played on team / speciality', 24: 'instance of', 25: 'participant', 26: 'has part', 27: 'composer', 28: 'sport', 29: 'located on terrain feature', 30: 'constellation', 31: 'place served by transport hub', 32: 'said to be the same as', 33: 'located in the administrative territorial entity', 34: 'field of work', 35: 'competition class', 36: 'country', 37: 'member of political party', 38: 'owned by', 39: 'distributor', 40: 'follows', 41: 'location', 42: 'operator', 43: 'original network', 44: 'screenwriter', 45: 'winner', 46: 'nominated for', 47: 'father', 48: 'platform', 49: 'tributary', 50: 'manufacturer', 51: 'notable work', 52: 'head of government', 53: 'league', 54: 'contains administrative territorial entity', 55: 'occupation', 56: 'performer', 57: 'architect', 58: 'position held', 59: 'movement', 60: 'mountain range', 61: 'religion', 62: 'location of formation', 63: 'spouse', 64: 'sibling', 65: 'military branch', 66: 'member of', 67: 'participant of', 68: 'child', 69: 'sports season of league or competition', 70: 'military rank', 71: 'mother', 72: 'director', 73: 'crosses', 74: 'licensed to broadcast to', 75: 'subsidiary', 76: 'after a work by', 77: 'applies to jurisdiction', 78: 'successful candidate', 79: 'voice type'}

SEMEVAL_IDX = {0: 'Other', 1: 'Instrument-Agency', 2: 'Entity-Origin', 3: 'Component-Whole', 4: 'Content-Container', 5: 'Member-Collection', 6: 'Product-Producer', 7: 'Message-Topic', 8: 'Cause-Effect', 9: 'Entity-Destination'}


def load_json(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)

import re

def calculate_label_accuracy(test_data, output_labels):
    label_correct_count = {label: 0 for label in set(test_data.values())}
    label_count = {label: 0 for label in set(test_data.values())}
    
    idx2label = FEWREL_IDX if len(test_data) == 5600 else SEMEVAL_IDX
    
    for sent in test_data:
        rel = test_data[sent]
        match = re.search(r'\d+', output_labels[sent])
        
        label_count[rel] += 1
        
        if match:  
            output_idx = match.group()  
            if int(output_idx) < len(idx2label):
                output = idx2label[int(output_idx)] 
            else:
                output = None
            
            if rel == output:
                label_correct_count[rel] += 1
        else:
            print(f"No match found for sentence: {sent}, treating as incorrect prediction.")

    label_acc = {label: count / label_count[label] if label_count[label] > 0 else 0
                 for label, count in label_correct_count.items()}
    
    overall_acc = sum(label_acc.values()) / len(label_acc) if len(label_acc) > 0 else 0
    
    return label_acc, overall_acc

def process_data(model, data_name):
    test_data_path = "../../../../data/FewRel/in-context/test_sent2rel.json" if data_name == "fewrel" else "../../../../data/SemEval/test.json"
    output_labels_path = f"./{data_name}-{model}.json"
    test_data = load_json(test_data_path)
    output_labels_temp = load_json(output_labels_path)
    output_labels = {sent: label.replace("relation: ", "") if "relation: " in label else label for sent, label in output_labels_temp.items()}

    label_accuracy, overall_accuracy = calculate_label_accuracy(test_data, output_labels)

    return label_accuracy, overall_accuracy

def generate_comparison_df(label_accuracy):
    df = pd.DataFrame({
        'Label': list(label_accuracy.keys()),
        'Accuracy': list(label_accuracy.values())
    })
    return df

res = {}
models = ["gemma", "llama"]
data_names = ["fewrel", "semeval"]

for model in models:
    for data_name in data_names:
        label_accuracy, overall_accuracy = process_data(
            model, data_name
        )

        comparison_df = generate_comparison_df(label_accuracy)

        comparison_df.to_csv(f'comparison_results_{model}_{data_name}.csv', index=False)

        res[f"{model}-{data_name}"] = overall_accuracy

with open('./icl-result.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(res.keys())
    writer.writerow(res.values())
