import os
import json
import warnings
import torch
from finetune import T5FineTuner
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoConfig,T5Tokenizer
import pandas as pd
from scipy.special import softmax
import numpy as np

warnings.filterwarnings("ignore",category=FutureWarning)

# optimize to support 观点聚合

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def revise_tuple(x,tokenizer_a,device,model_a,key_ls,tokenizer_b,model_b,config_b):
    features = tokenizer_a(x[0], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        scores = model_a(**features).logits
        labels = [key_ls[score_max] for score_max in scores.argmax(dim=1)]

    # get sentiment
    text = x[0]
    text = preprocess(text)
    encoded_input = tokenizer_b(text, return_tensors='pt').to(device)
    output = model_b(**encoded_input)
    scores = output[0][0].to(torch.device('cpu')).detach().numpy()
    scores = softmax(scores)
    res = config_b.id2label[np.argsort(scores)[-1]]

    # get final result
    result = (labels[0], res, x[1])
    return result


def model_fn(model_dir):
    """
    Load the model for inference
    """

    num_gpus = os.environ['num_gpus'] if ('num_gpus' in os.environ) else 0
    if(num_gpus > 0):
        device = torch.device(f'cuda:{0}')
    else:
        device = torch.device('cpu')

    saved_model_dir = '/opt/ml/model'
    all_checkpoints = []
    for f in os.listdir(saved_model_dir):
        file_name = os.path.join(saved_model_dir, f)
        if 'cktepoch' in file_name:
            all_checkpoints.append(file_name)
    print ("all checkpoints: ", all_checkpoints)

    checkpoint = os.path.join(saved_model_dir,all_checkpoints[-1])

    model_ckpt = torch.load(checkpoint, map_location=device)
    model = T5FineTuner(model_ckpt['hyper_parameters'])
    model.load_state_dict(model_ckpt['state_dict'])
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    model.model.to(device)
    model.model.eval()

    model_dict = {'model': model, 'tokenizer':tokenizer}

    # load two other models for revise
    model_a = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-distilroberta-base').to(device)
    tokenizer_a = AutoTokenizer.from_pretrained('cross-encoder/nli-distilroberta-base')
    model_a.eval()

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer_b = AutoTokenizer.from_pretrained(MODEL)
    config_b = AutoConfig.from_pretrained(MODEL)
    # PT
    model_b = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
    model_b.eval()

    df_keys = pd.read_csv('key.csv')
    key_ls = df_keys[df_keys['cate'] == x[1]]['normed_word'].tolist()

    model_dict = {'model': model,
                  'tokenizer': tokenizer,
                  'model_a':model_a,
                  'tokenizer_a':tokenizer_a,
                  'config_b':config_b,
                  'tokenizer_b':tokenizer_b,
                  'model_b':model_b,
                  'key_ls': key_ls
                  }
    
    return model_dict

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    tokenizer = model['tokenizer']
    model = model['model']

    data = input_data['inputs']
    max_seq_length=512

    num_gpus = os.environ['num_gpus'] if ('num_gpus' in os.environ) else 0
    if(num_gpus > 0):
        device = torch.device(f'cuda:{0}')
    else:
        device = torch.device('cpu')

    inputs = tokenizer(
              data, max_length=max_seq_length, pad_to_max_length=True, truncation=True,
              return_tensors="pt",
            )
    outs = model.model.generate(input_ids=inputs["input_ids"].to(device), 
                                    attention_mask=inputs["attention_mask"].to(device), 
                                    max_length=1024)
    dec=tokenizer.decode(outs[0], skip_special_tokens=True)

    revised_res = []
    for x in dec:
        revised_res.append(revise_tuple(x, model['tokenizer_a'], device, model['model_a'], model['key_ls'], model['tokenizer_b'], model['model_b'], model['config_b']))

    result =  {
        'result': dec,
        'revised_res':revised_res
    }

    return result

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input
    """
    
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        request = request_body

    return request

def output_fn(prediction, response_content_type):
    """
    Serialize and prepare the prediction output
    """
    
    if response_content_type == "application/json":
        response = json.dumps(prediction)
    else:
        response = str(prediction)

    return response