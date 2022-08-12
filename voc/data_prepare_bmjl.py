# This file process data for brat label result

#!pip install mendelai-brat-parser
import pandas as pd
import os
import argparse
from brat_parser import get_entities_relations_attributes_groups

def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--label_dir", default='../voc/review', type=str, required=True,
                        help="path contains brat label file")
    args = parser.parse_args()
    return args

def process_att(attributes):
    #attribute
    id_ls = []
    type_ls = []
    target_ls = []
    values_ls = []
    for i in range(len(list(attributes.values()))):
        id_ls.append(list(attributes.values())[i].id)
        type_ls.append(list(attributes.values())[i].type)
        target_ls.append(list(attributes.values())[i].target)
        values_ls.append(list(attributes.values())[i].values)

    df_attribute = pd.DataFrame({'id':id_ls,'type':type_ls,'target':target_ls,'values':values_ls})
    return df_attribute

def process_entities(entities):
    # attribute
    id_ls = []
    type_ls = []
    span_ls = []
    text_ls = []
    for i in range(len(list(entities.values()))):
        id_ls.append(list(entities.values())[i].id)
        type_ls.append(list(entities.values())[i].type)
        span_ls.append(list(entities.values())[i].span)
        text_ls.append(list(entities.values())[i].text)

    df_entities = pd.DataFrame({'id': id_ls, 'type': type_ls, 'span': span_ls, 'text': text_ls})
    return df_entities

def process_relations(relations):
    # attribute
    opinipn_ls = []
    aspect_ls = []

    for i in range(len(list(relations.values()))):
        opinipn_ls.append(list(relations.values())[i].subj)
        aspect_ls.append(list(relations.values())[i].obj)

    df_relations = pd.DataFrame({'opionion': opinipn_ls, 'aspect': aspect_ls})
    return df_relations

def process_brat(ann_file):
    #load file
    entities, relations, attributes, groups = get_entities_relations_attributes_groups(ann_file)
    df_attribute = process_att(attributes)
    df_entities = process_entities(entities)
    #df_relations = process_relations(relations)

    #merge
    # aspect join attr
    df_aspect_att = pd.merge(df_entities, df_attribute, left_on='id', right_on='target', how='left')
    return df_aspect_att


def process_text(text_file):
    # read txt file
    with open(text_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
    text = []
    sent_num = []
    sent_start = []
    sent_end = []
    j = 0
    sent_len = []
    for i in range(len(data)):
        sent_num.append(i)
        text.append(data[i])
        sent_start.append(j)
        sent_end.append(j + len(data[i]))
        sent_len.append(len(data[i]))
        j = j + len(data[i])

    df_data = pd.DataFrame({'sent_num': sent_num, 'text': text, 'sent_start': sent_start, 'sent_end': sent_end, 'sent_len': sent_len})
    return df_data

def judge_sen_number(x,df_data):
    '''
    判断如果(a,b)属于(c,d),则是第n句
    '''
    sent_num = df_data['sent_num']
    for i in range(len(sent_num)):
        start = df_data['sent_start'][i]
        end = df_data['sent_end'][i]
        if x>start:
            if x<end:
                return i
            else:
                continue
        else:
            return


def format_absa(aspect, category, opionion, sentiment):
    '''
    aspect It
    category (product,)
    opionion not work
    sentiment (0,)
    '''
    sent_dict = {'0': 'negative', '1': 'neural', '2': 'positive'}
    #print ("sentiment: ",sentiment)
    try:
        return (aspect, category[0], opionion, sent_dict[str(sentiment[0])])
    except:
        return

def format_aspect_category(aspect, category):
    '''
    aspect It
    category (product,)
    '''
    try:
        return (aspect, category[0])
    except:
        return


#merge back to data
def label_marge(df_data,df_res_four_labels):
    res = []
    #print ("type(df_res_four_labels['sent_number']),",df_res_four_labels[df_res_four_labels['sent_number']==0]['label'])
    for i in range(len(df_data)):
        res.append(list(df_res_four_labels[df_res_four_labels['sent_number']==i]['label']))
    return res

def process_single_file(text_file,ann_file,task):
    df_data = process_text(text_file)
    df_res = process_brat(ann_file)
    # add back sent_num
    df_res['start_idx'] = df_res['span'].map(lambda x: x[0][0])
    df_res['sent_number'] = df_res['start_idx'].map(lambda x: judge_sen_number(x, df_data))

    #print ("<<<< df_res.head()", df_res.head())
    df_res_four_labels = df_res[df_res['type_y']=='Category']
    df_res_four_labels['label'] = df_res_four_labels.apply(
            lambda row: format_aspect_category(row['text'], row['values']), axis=1)

    #filter out
    df_res_four_labels = df_res_four_labels[df_res_four_labels['label']!='']
    #print ("<<<<",df_res_four_labels.head())
    
    df_data['label'] = label_marge(df_data, df_res_four_labels)
    #print("<<<<",df_data.head())

    return df_data

def extract_label(jsonObj):
    jsonObj['label_tag'] = jsonObj['label'].map(lambda x:','.join(convert_label(x)))
    #map the tag list into single lines
    df=jsonObj.drop('label_tag', axis=1).join(jsonObj['label_tag'].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('tag'))

    #write the tag list
    a_list = list(df['tag'].unique())
    print ("a_list:", a_list)
    return a_list
    
def convert_label(x):
    res = []
    for i in x:
        res.append(i[1])

    return res

def process_all(label_dir,output_csv,task):
    #process over all files
    files = os.listdir(label_dir)
    ann_files = [i for i in files if '.ann' in i]
    res = pd.DataFrame({})
    for i in ann_files:
        txt_file = i.replace('.ann','.txt')
        try:
            df = process_single_file(os.path.join(label_dir,txt_file),os.path.join(label_dir,i),task)
            res = res.append(df)
            
        except:
            continue
            
    # write out label.txt 
    a_list = extract_label(res)
    #print (",".join(a_list))
    with open('tag.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % tag for tag in a_list)
    #write out to txt_files
    res.to_csv(output_csv)
    print ("finished process, result save to: ", output_csv)
    return res

def main(label_dir):
    #print ("process for (aspect,category,opinion,sentiment)")
    #df1 = process_all(label_dir, os.path.join(label_dir,'absa.csv'),task='absa')
    print("process for (aspect, category)")
    df2 = process_all(label_dir, os.path.join(label_dir,'aspect_category.csv'),task='aspect_category')
    #write out to train/test/val


if __name__ == '__main__':
    #run
    args = init_args()
    main(args.label_dir)
