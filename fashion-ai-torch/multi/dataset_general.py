import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class AttributesDataset():
    def __init__(self, annotation_path, keylist):
        self.res_labels = {}
        self.id_to_name = {}
        self.name_to_id = {}
        self.label_len = {}

        #self.keys = ['顏色(Color)','圖案(Pattern Type)','風格(Style)']
        self.keys = keylist

        for i in self.keys:
            self.res_labels[i] = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for j in self.keys:
                    self.res_labels[j].append(row[j])

        for i in self.keys:
            self.res_labels[i] = np.unique(self.res_labels[i])
            self.label_len[i] = len((self.res_labels[i]))
            self.id_to_name[i] = dict(zip(range(len(self.res_labels[i])), self.res_labels[i]))
            self.name_to_id[i] = dict(zip(self.res_labels[i],range(len(self.res_labels[i]))))

        self.feature_dict = {}
        for i in self.keys:
            self.feature_dict[i] = self.label_len[i]

class FashionDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.labels = {}

        for i in self.attr.keys:
            self.labels[i] = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                for i in self.attr.keys:
                    #print ("<<< key:", i)
                    #print("<<< row[i]:", row[i])

                    #print ('<<< self.attr.name_to_id[i]', self.attr.name_to_id[i])
                    self.labels[i].append(self.attr.name_to_id[i][row[i]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # read image
        img = Image.open(img_path)

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels

        label_dict = {}
        for i in self.attr.keys:
            label_dict[i+'_labels'] = self.labels[i][idx]

        dict_data = {
            'img': img,
            'labels': label_dict
        }
        return dict_data