import os
import json
import warnings
import torch
from finetune import T5FineTuner
from transformers import (
    T5Tokenizer
)
warnings.filterwarnings("ignore",category=FutureWarning)

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

    result =  {
        'result': dec
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