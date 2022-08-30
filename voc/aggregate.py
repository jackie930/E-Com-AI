# pip install transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def init_model_agg():
    classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-distilroberta-base')
    return classifier


def init_model_sentiment():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model.eval()
    return tokenizer, config, model


def main(x):
    # x = ("fit well", "size")
    df_keys = pd.read_csv('key.csv')
    key_ls = df_keys[df_keys['cate'] == x[1]]['normed_word'].tolist()

    # get agg
    classifier = init_model_agg()
    sent = x[0]
    candidate_labels = key_ls
    res = classifier(sent, candidate_labels)
    labels = res['labels']
    scores = res['scores']
    if scores[0] > 0.2:
        labels = labels
    else:
        labels = [sent]

    # get sentiment
    tokenizer, config, model = init_model_sentiment()
    text = x[0]
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    output = model(**encoded_input)
    scores = output[0][0].to(torch.device('cpu')).detach().numpy()
    scores = softmax(scores)
    res = config.id2label[np.argsort(scores)[-1]]

    # get final result
    result = (labels[0], res, x[1])

    print("input: ", x)
    print("finish process, results: ", result)
    return result


if __name__ == '__main__':
    x = ("fit well", "size")
    main(x)