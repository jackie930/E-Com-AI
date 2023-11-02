import argparse
import json
import os

def data_process(path):
    with open(path, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()
    return raw_examples

def write_back(raw_examples):
    res = []

    for i in range(len(raw_examples)):
        x1 = json.loads(raw_examples[i])
        dict_res = {'id': x1['id'],
                    'text': x1['data'],
                    'entities': x1['label']['entities'],
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




