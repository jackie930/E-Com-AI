#!pip install openpyxl
import argparse
import pandas as pd
import os
import json
import shutil
from sklearn.model_selection import train_test_split


def get_key_list(x):
    # get key dictionary
    t = json.loads(x)
    res = [i for i in list(t.values())]
    res = [list(i.keys())[0] for i in res]
    return res


def get_keys(df):
    #return keys
    lst = list(df['feature_dict'])
    myList = [x for j in lst for x in j]
    res = list(set(myList))
    # res_str = ','.join(res)
    return res


def get_key_value(x, i):
    #maake key columns
    t = json.loads(x)

    res = [i for i in list(t.values())]
    keys = [list(i.keys())[0] for i in res]
    values = [list(i.values())[0] for i in res]
    dict_res = dict(zip(keys, values))
    if i in dict_res.keys():
        return dict_res[i]
    else:
        return 'other'

def test_path(x, category):
    #test the path if exists
    root_path = os.path.join('/home/ec2-user/SageMaker/data_0731', category)
    img_name = os.path.join(root_path, str(x) + '.png')
    # print ('img_name',img_name)
    if os.path.exists(img_name):
        return img_name
    else:
        return 'none'

def self_mkdir(folder):
    isExists = os.path.exists(folder)
    if not isExists:
        os.makedirs(folder)
        print('path of %s is build' % (folder))
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)
        print('path of %s already exist and rebuild' % (folder))

def get_data(path, category, output_dir):
    df = pd.read_excel(path, engine="openpyxl")
    df = df[df['creg'] == category]
    # df['feature_len'] = df['data'].map(lambda x: get_feature_len(x))
    # leng = max(df['feature_len'])
    df['feature_dict'] = df['data'].map(lambda x: get_key_list(x))
    res_keys = get_keys(df)
    print("<<< predict for keys: ", ','.join(res_keys))

    for i in res_keys:
        df[i] = df['data'].map(lambda x: get_key_value(x, i))

    # repath
    df['image_path'] = df['md5_url'].map(lambda x: test_path(x, category))
    df = df[df['image_path'] != 'none']

    # make dir if not exist
    self_mkdir(output_dir)
    # save data
    df[res_keys].to_csv(os.path.join(output_dir, 'total.csv'), index=False)
    train, test = train_test_split(df, test_size=0.25, random_state=0)
    train.to_csv(os.path.join(output_dir, 'train.csv'))
    test.to_csv(os.path.join(output_dir, 'test.csv'))
    print("train size {}, test size{}".format(train.shape, test.shape))
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data and save for the dataset')
    parser.add_argument('--input', type=str, default='./data_0731/shein_info.xlsx', required=True, help="Path to the dataset")
    parser.add_argument('--output', type=str, default='./model_data', required=True, help="Path to the working folder")
    parser.add_argument('--category', type=str, default='Women-Tops,-Blouses-Tee', required=True, help="Path to the working folder")

    args = parser.parse_args()

    df = get_data(args.input, category=args.category, output_dir=args.output)

