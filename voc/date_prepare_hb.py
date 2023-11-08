import pandas as pd
import numpy as np
import os
import shutil


# preprocess data
def write_txt(df, path):
    '''
    write back to txt
    '''
    # output txt file
    df = df.reset_index()
    with open(path, 'a') as f:
        for i in range(len(df)):
            f.write("{} #### {}".format(df.loc[i, 'review_content'].strip(), df.loc[i, 'label']))
            f.write('\n')

def get_key():
    category_list = pd.read_excel('../../data/GPT预打标20231023.xlsx', sheet_name='标签释义1012',header=1)
    key = list(category_list['三级指标 (En)'])
    value = list(category_list['三级指标 GPT翻译'])
    key = [x.capitalize() for x in key]
    value = [x.capitalize() for x in value]
    keydict = dict(zip(key, value))
    return keydict, value, key


def convert_key(x, key_dict, values, keys):
    if x in values:
        return x
    elif x in keys:
        return key_dict[x]
    else:
        return 'wrong'


def mkdir_rm(folder):
    '''
    make directory if not exists
    '''
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.mkdir(folder)
    print("<< path valid!")

def get_label(row):
    try:
        if pd.isna(row['Remark']):
            return (str(row['aspect_term']),str(row['aspect_category']),str(row['opinion_term']),str(row['sentiment_polarity']))
        elif row['Remark']=='改':
            return (str(row['x1']),str(row['x2']),str(row['x3']),str(row['x4']))
        elif row['Remark']=='修改':
            return (str(row['x1']),str(row['x2']),str(row['x3']),str(row['x4']))
    except:
        return (str(row['aspect_term']),str(row['aspect_category']),str(row['opinion_term']),str(row['sentiment_polarity']))

def get_data(type,excelname,sheetname=None):
    if type == 'direct':
        jsonObj = pd.read_excel(excelname)
    elif type == 'subsheet':
        jsonObj = pd.read_excel(excelname, sheet_name=sheetname)
    #revise content
    jsonObj['review_content'] = jsonObj['review_content'].fillna(method='ffill')
    jsonObj = jsonObj[-jsonObj['review_content'].isnull()]
    jsonObj['review_content'] = jsonObj['review_content'].apply(lambda x: x.replace('\n', ' '))

    #get the right keys
    # readjust key
    key_dict, values, keys = get_key()
    # capitalize
    jsonObj['aspect_category'] = jsonObj['aspect_category'].map(lambda x: str(x).capitalize())
    # map
    jsonObj['aspect_category'] = jsonObj['aspect_category'].map(lambda x: convert_key(x, key_dict, values, keys))
    jsonObj = jsonObj[jsonObj['aspect_category'] != 'wrong']

    if 'x2' in jsonObj.columns:
        jsonObj['x2'] = jsonObj['x2'].map(lambda x: str(x).capitalize())
        # map
        jsonObj['x2'] = jsonObj['x2'].map(lambda x: convert_key(x, key_dict, values, keys))

    #todo: revise category keys
    revise_dict = pd.read_excel('../../data/keys.xlsx')
    revise_dict['key'] = revise_dict['key'].map(lambda x: str(x).capitalize())
    revise_dict['values'] = revise_dict['values'].map(lambda x: str(x).capitalize())
    dictionary = revise_dict.set_index('key')['values'].to_dict()

    if 'x2' in jsonObj.columns:
        jsonObj['x2'] = jsonObj['x2'].map(lambda x: dictionary[x] if x in dictionary.keys() else x)
    jsonObj['aspect_category'] = jsonObj['aspect_category'].map(lambda x: dictionary[x] if x in dictionary.keys() else x)

    #map sentiment
    if 'x4' in jsonObj.columns:
        jsonObj['x4'] = jsonObj['x4'].map(lambda x: 'Positive' if str(x).capitalize()=='Neutral' else x)
    jsonObj['sentiment_polarity'] = jsonObj['sentiment_polarity'].map(lambda x: 'Positive' if str(x).capitalize()=='Neutral' else x)

    # generate label
    jsonObj['label'] = jsonObj.apply(lambda row: get_label(row), axis=1)

    jsonObj = jsonObj[['review_content', 'label']]
    jsonObj = jsonObj[-jsonObj['label'].isnull()]

    # agg label
    jsonObj_grouped = jsonObj.groupby('review_content')['label'].apply(list).reset_index()
    return jsonObj_grouped


def preprocess_data(output_path, over_sample=True):
    # remove & remake the output folder
    mkdir_rm(output_path)

    # get data
    dfv1 = get_data('subsheet','../../data/GPT预打标20231023.xlsx','US-10-13-（刘莎-展诚-Nick）')
    dfv2 = get_data('subsheet','../../data/GPT预打标20231023.xlsx','3星及以下（145）-产品核验-标注-1007')
    dfv3 = get_data('subsheet','../../data/Suzy人工打标.xlsx','Sheet1')

    result = pd.concat([dfv1, dfv2, dfv3])

    # train/test/val split
    train, validate, test = np.split(result.sample(frac=1), [int(.8 * len(result)), int(.9 * len(result))])

    # concat 2000
    df1 = get_data('direct','../../data/GPT打标（带描述）-Jackie.xlsx')
    df2 = get_data('direct','/../../data/claude100条.xlsx')
    train = pd.concat([df1, df2, train])

    print("training size: ", train.shape)
    print("test size: ", test.shape)
    print("validate size: ", validate.shape)

    # write train/test/dev
    write_txt(train, os.path.join(output_path, 'train.txt'))
    write_txt(test, os.path.join(output_path, 'test.txt'))
    write_txt(validate, os.path.join(output_path, 'dev.txt'))
    print("<<<finish data preparing!")


output_path = './data/tasd/huabaoall'
preprocess_data(output_path, over_sample=False)