from pprint import pprint
from paddlenlp import Taskflow
import pandas as pd
from model import UIE
from tqdm import tqdm
import json
import os

def data_process(folder):
    df_source = pd.read_csv(os.path.join(folder, 'detials_connectivity.csv'))
    df_label = pd.read_csv(os.path.join(folder, 'connectivity_info.csv'))
    return df_source, df_label

def init_flow():
    schema = ['asin',
              'design for Device',
              'Hub/Dock',
              'Number of Ports',  # 接口数
              'usb transfer speed',  # USB接口传输速度
              'SD transfer speed',  # SD卡传输速度
              'contain HDMI hub',  # 含有HDMI接口
              'contain VGA hub',  # 含有VGA接口
              ]  # Define the schema for entity extraction
    ie = Taskflow("information_extraction", model='uie-base-en', schema=schema, home_path='./uie-base-en')
    return ie

def _save_examples(save_dir, file_name, examples):
    count = 0
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    print("Save %d examples to %s." % (count, save_path))

if __name__ == '__main__':
    #model = UIE.from_pretrained('taskflow/information_extraction/uie-en')
    #model = UIE.from_pretrained('/Users/liujunyi/Documents/github/E-Com-AI/cpv/uie-base-en/taskflow/information_extraction/uie-base-en')
    df_source, df_label = data_process('./data')
    ie = init_flow()
    examples = []
    for i in tqdm(range(2)):
        asin = df_source['asin'][i]
        text = df_source['detail_json'][i]
        print ("text: ", text)
        res = ie(text)
        print ("<<< res: ", res[0])
        try:
            for i in res[0].keys():
                print ("<<< key", i)
                for j in res[0][i]:
                    if j['probability']>0.7:
                        print("<<< prob: ", res[0][i][0]['probability'])

                        dict_res = {}
                        dict_res['content'] = text
                        dict_res['prompt'] = i
                        word_freq_dict = {key: value for key, value \
                                          in j.items() \
                                          if key != "probability"}
                        dict_res['result_list'] = word_freq_dict
                        print ("<<< dict_res: ", dict_res)

                        if i=='contain HDMI hub':
                            label = df_label[df_label['asin']==asin]['含有HDMI接口']
                            print("hdmi label: ", label)
                            if label:
                                examples.append(dict_res)

                        elif i=='contain VGA hub':
                            label = df_label[df_label['asin'] == asin]['含有VGA接口']
                            print("VGA label: ", label)
                            if label:
                                examples.append(dict_res)

                        else:
                            examples.append(dict_res)
        except:
            continue

    _save_examples('./train', 'all.txt', examples)



