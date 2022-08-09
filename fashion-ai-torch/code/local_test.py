import torch
from multi.model_general import MultiOutputModel
from multi.dataset_general import FashionDataset, AttributesDataset
import torchvision.transforms as transforms
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attributes = AttributesDataset('./model_data/total.csv')
model = MultiOutputModel(feature_dict=attributes.feature_dict).to(device)

name = './checkpoints/2022-08-08_06-14/checkpoint-000050.pth'
print('Restoring checkpoint: ')
model.load_state_dict(torch.load(name, map_location=device))
model.eval()

if __name__ == '__main__':
    img = Image.open('data_0731/Women-Tops,-Blouses-Tee/00101e913bd577125e5509b78117b873.html.png')
    #如果是四通道， 处理为三通道
    #img = img.convert("RGB")

    val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    x = val_transform(img)
    inputs = x.unsqueeze(0)
    output = model(inputs.to(device))

    return_res = {}
    for i in list(output.keys()):
        idx = output[i].max(1)[1].item()
        return_res[i] = attributes.id_to_name[i][idx]

    print (return_res)