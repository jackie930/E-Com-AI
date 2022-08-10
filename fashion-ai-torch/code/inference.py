import os
import json
import warnings
import torch

import torchvision.transforms as transforms

from PIL import Image
import io
from model_general import MultiOutputModel

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
        if 'checkpoint' in file_name:
            all_checkpoints.append(file_name)
    print ("all checkpoints: ", all_checkpoints)

    checkpoint = os.path.join(saved_model_dir,all_checkpoints[-1])
    model_ckpt = torch.load(checkpoint, map_location=device)

    #res_dict = {}
    #for k, v in model_ckpt.items():
     #   if k[:5] == 'tasks':
      #      res_dict[k.split(".")[1]] = len(v)

    model = MultiOutputModel(feature_dict=model_ckpt['feature_dict']).to(device)
    model.load_state_dict(model_ckpt['model_state_dict'])
    model.eval()

    model_dict = {'model': model,
                  'id_to_name':model_ckpt['feature_dict_map']}
    
    return model_dict

def load_from_bytearray(request_body):
    image_as_bytes = io.BytesIO(request_body)
    image = Image.open(image_as_bytes)
    #todo: check 3 bands
    #todo: instead of simple resize, consider filling
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    img_size = [1785, 1340]
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    x = val_transform(image)

    inputs = x.unsqueeze(0)

    return inputs

def predict_fn(input_data, model):
    """
    Apply model to the incoming request
    """

    num_gpus = os.environ['num_gpus'] if ('num_gpus' in os.environ) else 0
    if(num_gpus > 0):
        device = torch.device(f'cuda:{0}')
    else:
        device = torch.device('cpu')

    output = model['model'](input_data.to(device))

    #map back
    return_res = {}
    for i in list(output.keys()):
        idx = output[i].max(1)[1].item()
        return_res[i] = model['id_to_name'][i][idx]

    result = {
        'result': return_res
    }

    return result

def input_fn(request_body, request_content_type):
    # if set content_type as "image/jpg" or "application/x-npy",
    # the input is also a python bytearray
    if request_content_type == "application/x-image":
        image_tensor = load_from_bytearray(request_body)
    else:
        print("not support this type yet")
        raise ValueError("not support this type yet")
    return image_tensor

def output_fn(predictions, response_content_type):
    return json.dumps(predictions)
