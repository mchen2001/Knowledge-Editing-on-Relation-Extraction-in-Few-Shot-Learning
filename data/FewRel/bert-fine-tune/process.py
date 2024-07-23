import json
import re
import os

os.chdir("./data/FewRel/bert-fine-tune")


with open('../relid2idx.json', 'r') as f:
    rel2idx = json.load(f)

def join_tokens(tokens):
    sentence = '[CLS] '
    for token in tokens:
        if token in ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']:
            sentence += f' {token}'
        elif token in [',', '.', '!', '?', ':', ';', ')']:
            sentence += token
        elif token in ['(']:
            sentence += f' {token}'
        else:
            sentence += f' {token}'
    sentence += f' [SEP]'
    return sentence.strip()


def find_sublist_index(lst, sublst):
    lst_lower = [x.lower() for x in lst]
    sublst_lower = [x.lower() for x in sublst]
    
    for i in range(len(lst_lower) - len(sublst_lower) + 1):
        if lst_lower[i:i+len(sublst_lower)] == sublst_lower:
            return i
    raise ValueError("Sublist not found")

# Function to add special tokens around head and tail entities
def add_special_tokens(example):
    tokens = example['tokens']
    head = re.findall(r'\w+|[^\w\s]+', example['head'])
    tail = re.findall(r'\w+|[^\w\s]+', example['tail'])
    
    try:
        head_start = find_sublist_index(tokens, head)
        tail_start = find_sublist_index(tokens, tail)
        head_end = head_start + len(head)
        tail_end = tail_start + len(tail)
    except ValueError:
        print(f"Entities not found in tokens: {example}")
        return example

    # Insert special tokens for head entity
    tokens.insert(head_start, '<SUBJ_START>')
    tokens.insert(head_end + 1, '<SUBJ_END>')
    
    if tail_start > head_end:
        tail_start += 2
        tail_end += 2
    
    # Insert special tokens for tail entity
    tokens.insert(tail_start, '<OBJ_START>')
    tokens.insert(tail_end + 1, '<OBJ_END>')
    
    example['tokens'] = join_tokens(tokens)
    return example

def main():
    for n in [2,4,8,16,32]:
        for i in range(10):
            process(f'../few-shot-sample/{n}/{i}.json', f'{n}/{i}.json')
    
    with open("../test_data.json", 'r') as f:
        test_datas = f.readlines()

    test_processed = []
    for test_data in test_datas:
        test_data = json.loads(test_data)
        test_data = add_special_tokens(test_data)
        test_data['relation'] = rel2idx[test_data['relation']]
        test_processed.append(test_data)
        
    with open("test_bert.json", 'w') as f:
        json.dump(test_processed, f, ensure_ascii=False, indent=4)


    print("Processing complete.")

def process(infile, outfile):
    with open(infile, 'r') as f:
        data = json.load(f)

    processed_data = []
    for example in data:
        example = add_special_tokens(example)
        example['relation'] = rel2idx[example['relation']]
        processed_data.append(example)

    with open(outfile, 'w') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()