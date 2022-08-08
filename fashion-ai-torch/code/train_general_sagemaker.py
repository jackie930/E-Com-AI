import argparse
import os
from datetime import datetime

if os.path.exists('./requirements.txt'):
    os.system('pip install -r requirements.txt')
else:
    os.system('pip install -r /opt/ml/model/code/requirements.txt')
os.system('pip install tensorboard')

import torch
import torchvision.transforms as transforms
from dataset_general import FashionDataset, AttributesDataset, mean, std
from model_general import MultiOutputModel
from test import calculate_metrics, validate, visualize_grid, calculate_metrics_general,validate_general
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from time import sleep
from tqdm import tqdm

#print ("<<<< list dir: ", os.listdir('/opt/ml/input/data/training'))

def get_cur_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model, name, epoch):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print('Saved checkpoint:', f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--attributes_file', default='/opt/ml/input/data/training/total.csv', type=str,
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cuda', help="Device: 'cuda' or 'cpu'")
    parser.add_argument('--sourcedir', type=str, default='/opt/ml/input/data/training', help="train/test data folder")
    parser.add_argument('--batch_size', type=int, default=8, help="batch size")
    parser.add_argument('--epoch', type=int, default=1, help="train epoch")
    parser.add_argument('--save_epoch', type=int, default=1, help="save every n epoch")
    parser.add_argument('--val_epoch', type=int, default=1, help="val every n epoch")
    parser.add_argument('--num_workers', type=int, default=1, help="num_workers")

    args = parser.parse_args()

    start_epoch = 1
    N_epochs = args.epoch
    batch_size = args.batch_size
    num_workers = args.num_workers  # number of processes to handle dataset loading
    print ("<<< num workers: ", args.num_workers )
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    # attributes variable contains labels for the categories in the dataset and mapping between string names and IDs
    attributes = AttributesDataset(args.attributes_file)

    img_size = [1785,1340]
    # specify image transforms for augmentation during training
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
        transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2),
                                shear=None, resample=False, fillcolor=(255, 255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # during validation we use only tensor and normalization transforms
    val_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = FashionDataset(os.path.join(args.sourcedir, 'train.csv'), attributes, train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_dataset = FashionDataset(os.path.join(args.sourcedir, 'test.csv'), attributes, val_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = MultiOutputModel(feature_dict=attributes.feature_dict).to(device)
    #print (summary(model,(3,244,244)))

    optimizer = torch.optim.Adam(model.parameters())

    logdir = os.path.join('./logs/', get_cur_time())
    savedir = '/opt/ml/model/'
    #savedir = os.path.join('./checkpoints/', get_cur_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    n_train_samples = len(train_dataloader)

    # Uncomment rows below to see example images with ground truth labels in val dataset and all the labels:
    # visualize_grid(model, val_dataloader, attributes, device, show_cn_matrices=False, show_images=True,
    #                checkpoint=None, show_gt=True)
    # print("\nAll gender labels:\n", attributes.gender_labels)
    # print("\nAll color labels:\n", attributes.color_labels)
    # print("\nAll article labels:\n", attributes.article_labels)

    print("Starting training ...")

    for epoch in range(start_epoch, N_epochs + 1):
        total_loss = 0

        acc = {}
        for i in attributes.feature_dict:
            acc[i] = 0


        #with tqdm(train_dataloader) as tepoch:
        tepoch = tqdm(train_dataloader)
        for batch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            optimizer.zero_grad()

            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(device) for t in target_labels}
            output = model(img.to(device))

            loss_train, losses_train = model.get_loss(output, target_labels)
            total_loss += loss_train.item()
            batch_accuracy = calculate_metrics_general(output, target_labels)

            for j in acc.keys():
                acc[j] += batch_accuracy[j]

            loss_train.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss_train.item())

            #tepoch.set_postfix(loss=loss_train.item(), accuracy=acc)
            sleep(0.1)


        print("epoch {:4d}, loss: {:.4f}, n_train_samples: {:4d}".format(
            epoch,
            total_loss / n_train_samples,
            n_train_samples))

        for i in attributes.feature_dict:
            print("epoch {:4d}, accuracy for {} is {}".format(epoch, i, acc[i] / n_train_samples))

        logger.add_scalar('train_loss', total_loss / n_train_samples, epoch)

        if epoch % args.val_epoch == 0:
            validate_general(model, val_dataloader, attributes, device)

        if epoch % args.save_epoch == 0:
            checkpoint_save(model, savedir, epoch)
