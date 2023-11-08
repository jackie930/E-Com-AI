from tqdm import tqdm
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def cosine_sim(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

def find_similar_x(str1, strlist):
    scores = []
    for i in strlist:
        score = cosine_sim(str1, i)
        scores.append(score)
    if max(scores)>0.5:
        idx = scores.index(max(scores))
    else:
        idx = -1
    return idx

def extract_spans_extraction(seq, label_len=2):
    extractions = []
    extractions_with_aspect = []
    all_pt = seq.split('; ')

    # print ("<<< all_pt", all_pt)
    for pt in all_pt:
        if label_len == 4:
            try:
                a, b, c, d = pt[1:-1].split(', ')
            except ValueError:
                continue
        extractions.append((a))
        extractions_with_aspect.append((a, b, c, d, a + ' ' + c))
    return extractions_with_aspect


def extract(preds, labels):
    same = []
    same_2 = []
    compre_str = [i[4] for i in preds]
    for i in range(len(labels)):
        target = labels[i]
        target_str = labels[i][4]
        similar_idx = find_similar_x(target_str, compre_str)
        if similar_idx == -1:
            same.append('none')
            same_2.append('none')
        else:
            same.append(preds[similar_idx][1])
            same_2.append(preds[similar_idx])
    return same, same_2

def main(df):
    df_all = pd.DataFrame({'sent': [],
                           'label_all': [],
                           'label': [],
                           'pred_cate': [],
                           'pred': []})

    for i in tqdm(range(len(df['outputs']))):
        preds = extract_spans_extraction(df['outputs'][i], 4)
        labels = extract_spans_extraction(df['targets'][i], 4)
        same_tag, same_pred = extract(preds, labels)
        str_sent = ' '.join(df['sents'][i])
        sents = [str_sent] * len(same_tag)

        df_res = pd.DataFrame({'sent': sents,
                               'label_all': labels,
                               'label': [i[1] for i in labels],
                               'pred_cate': same_tag,
                               'pred': same_pred})

        df_all = pd.concat([df_all, df_res])
    return df_all

df = pd.read_pickle('results-tasd-huabao1023-extraction-allres.pickle')
res = main(df)