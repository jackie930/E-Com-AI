import argparse
import json
import os

def data_process(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()
    return raw_examples

def init_key():
    test_keys = ['鞋型（Type）',
     '鞋跟高度（Heel Height）',
     '款式（Pattern）',
     '鞋跟类型（Heel Type）',
     '鞋头类型（Toe Style）']
    test_values = ['Shoe Type',
     'Shoe Heel Height',
     'Shoe Pattern',
     'Shoe Heel Type',
     'Shoe Toe Style']

    res = {}
    for key in test_keys:
        for value in test_values:
            res[key] = value
            test_values.remove(value)
            break
    return res

def convert(x,key_dict):
    for i in x:
        i['label'] = key_dict[i['label']] 
    return x

def write_back(raw_examples):
    res = []
    
    key_dict = init_key()
    for i in range(len(raw_examples)):
        x1 = json.loads(raw_examples[i])
        dict_res = {'id': x1['id'],
                    'text': x1['data'],
                    'entities': convert(x1['label']['entities'],key_dict),
                    'relations': []}
        res.append(dict_res)
    return res

def save_single_file(input_path, output_folder):
    raw_examples = data_process(input_path)
    res = write_back(raw_examples)
    # write back
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(os.path.join(output_folder, input_path.split('/')[-1]), 'w', encoding="utf-8") as json_file:
        for i in res:
            json_file.write(json.dumps(i, ensure_ascii=False) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prepare train data for shulex')

    parser.add_argument('--input_path', type=str, default='./', help="input file or input folder path")
    parser.add_argument('--output_folder', type=str, default='./', help="output folder path")

    args = parser.parse_args()

    save_single_file(args.input_path, args.output_folder)




