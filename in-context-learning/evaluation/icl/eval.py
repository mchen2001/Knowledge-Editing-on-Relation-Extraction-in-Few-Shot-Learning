import json
import pandas as pd
import csv

def load_json(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        return json.load(f)

def calculate_label_accuracy(test_data, output_labels):
    label_correct_count = {label:0 for label in set(test_data.values())}
    label_count = {label:0 for label in set(test_data.values())}
    for sent in test_data:
        rel = test_data[sent]
        output = output_labels[sent]
        label_count[rel] += 1
        if rel.lower() == output:
            label_correct_count[rel] += 1
    label_acc = {label:count/label_count[label] for label,count in label_correct_count.items()}
    return label_acc, sum(label_acc.values()) / len(label_acc)

def clean_label(label):
    label = label.lower()
    label = label.replace("relation: ", "")
    label = label.replace("'", "")
    label = label.replace("\"", "")
    label = label.replace("_", " ")
    return label

def process_data(M, model, data_name, fine_tune_data):
    test_data_path = "../../../data/FewRel/in-context/test_sent2rel.json" if data_name == "fewrel" else "../../../data/SemEval/test.json"
    output_labels_path = f"./{data_name}/{fine_tune_data}/{data_name}-{M}-{model}.json"
    test_data = load_json(test_data_path)
    output_labels_temp = load_json(output_labels_path)
    output_labels = {sent: clean_label(label) for sent, label in output_labels_temp.items()}

    label_accuracy, overall_accuracy = calculate_label_accuracy(test_data, output_labels)

    return label_accuracy, overall_accuracy

def generate_comparison_df(label_accuracy):
    df = pd.DataFrame({
        'Label': list(label_accuracy.keys()),
        'Accuracy': list(label_accuracy.values())
    })
    return df

res = {}
Ms = [4,32,128]
models = ["gemma", "llama"]
data_names = ["fewrel", "semeval"]
fine_tune_data = {"fewrel": "semeval", "semeval": "fewrel128"}

for M in Ms:
    for model in models:
        for data_name in data_names:
            label_accuracy, overall_accuracy = process_data(
                M, model, data_name, fine_tune_data[data_name]
            )

            comparison_df = generate_comparison_df(label_accuracy)

            comparison_df.to_csv(f'./{data_name}/{fine_tune_data[data_name]}/comparison_results_{M}_{model}_{data_name}.csv', index=False)
            res[f"{M}-{model}-{data_name}-{fine_tune_data[data_name]}"] = overall_accuracy

with open('./icl-result.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(res.keys())
    writer.writerow(res.values())