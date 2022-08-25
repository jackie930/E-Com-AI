#pip install transformers
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax

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
    model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-distilroberta-base').to(device)
    tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-distilroberta-base')
    model.eval()
    return tokenizer, model

def init_model_sentiment():
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model.eval()
    return tokenizer,config,model

def main(x):
    #x = ("fit well", "size")
    df_keys = pd.read_csv('key.csv')
    key_ls = df_keys[df_keys['cate'] == x[1]]['normed_word'].tolist()

    # get agg
    tokenizer, model = init_model_agg()
    features = tokenizer(x[0], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        scores = model(**features).logits
        labels = [key_ls[score_max] for score_max in scores.argmax(dim=1)]

    #get sentiment
    tokenizer, config, model = init_model_sentiment()
    text = x[0]
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt').to(device)
    output = model(**encoded_input)
    scores = output[0][0].to(torch.device('cpu')).detach().numpy()
    scores = softmax(scores)
    res = config.id2label[np.argsort(scores)[-1]]

    #get final result
    result = (labels[0],res,x[1])

    print ("input: ", x)
    print ("finish process, results: ", result)
    return result

if __name__ == '__main__':
    x = ("fit well", "size")
    main(x)