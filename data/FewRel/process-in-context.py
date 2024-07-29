import json
import os

os.chdir("./data/FewRel/")


def transform_sentence(data, idx2relid, pid2name):
    transformed_data = []
    for item in data:
        sentence = item['tokens']
        relation = item['relation']
        relation = idx2relid[str(relation)]
        relation = pid2name[relation][0]

        sentence = sentence.replace('<SUBJ_START>', '<entity1>').replace('<SUBJ_END>', '</entity1>')
        sentence = sentence.replace('<OBJ_START>', '<entity2>').replace('<OBJ_END>', '</entity2>')
        
        sentence = sentence.replace('[CLS]', '').replace('[SEP]', '').strip()
        
        new_sentence = f"{sentence}\nrelation: {relation}"
        transformed_data.append(new_sentence)
    
    return transformed_data

def main():
    with open("./idx2relid.json", "r") as f:
        idx2relid = json.load(f)

    with open("./pid2name.json", "r") as f:
        pid2name = json.load(f)
    
    for i in [2,4,8,16,32,64,128,256,512]:
        with open(f"./bert-fine-tune/{i}/0.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        transformed_data = transform_sentence(data, idx2relid, pid2name)
        with open(f'./in-context/{i}.json', 'w', encoding="utf-8") as f:
            json.dump(transformed_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()