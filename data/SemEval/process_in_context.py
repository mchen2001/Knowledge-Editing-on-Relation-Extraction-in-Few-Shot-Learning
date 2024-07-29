import jsonlines
import json
import os

os.chdir("./data/SemEval/")

def load_json(file_path):
    data = []

    with jsonlines.open(file_path, mode='r') as reader:
        for obj in reader:
            data.append(obj)
    return data


def transform_sentence(data):
    transformed_data = []
    for item in data:
        sentence = item['sentence']
        relation = item['relation']
        
        sentence = sentence.replace('<SUBJ_START>', '<entity1>').replace('<SUBJ_END>', '</entity1>')
        sentence = sentence.replace('<OBJ_START>', '<entity2>').replace('<OBJ_END>', '</entity2>')
        
        sentence = sentence.replace('[CLS]', '').replace('[SEP]', '').strip()
        
        new_sentence = f"{sentence}\nrelation: {relation}"
        transformed_data.append(new_sentence)
    
    return transformed_data


def main():
    for i in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        file_path = f'./few_shot_samples/{i}.json'
        data = load_json(file_path)
        print(len(data))

        transformed_data = transform_sentence(data)

        # for sentence in transformed_data:
        #     print(sentence)

        with open(f'in-context/{i}.json', 'w') as f:
            json.dump(transformed_data, f, indent=4)

if __name__ == '__main__':
    main()