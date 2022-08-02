import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class AttributesDataset():
    def __init__(self, annotation_path):
        #todo: update by num class
        color_labels = []
        pattern_labels = []
        style_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                color_labels.append(row['顏色(Color)'])
                pattern_labels.append(row['圖案(Pattern Type)'])
                style_labels.append(row['風格(Style)'])

        self.color_labels = np.unique(color_labels)
        self.pattern_labels = np.unique(pattern_labels)
        self.style_labels = np.unique(style_labels)

        self.num_colors = len(self.color_labels)
        self.num_patterns = len(self.pattern_labels)
        self.num_styles = len(self.style_labels)

        self.color_id_to_name = dict(zip(range(len(self.color_labels)), self.color_labels))
        self.color_name_to_id = dict(zip(self.color_labels, range(len(self.color_labels))))

        self.pattern_id_to_name = dict(zip(range(len(self.pattern_labels)), self.pattern_labels))
        self.pattern_name_to_id = dict(zip(self.pattern_labels, range(len(self.pattern_labels))))

        self.style_id_to_name = dict(zip(range(len(self.style_labels)), self.style_labels))
        self.style_name_to_id = dict(zip(self.style_labels, range(len(self.style_labels))))


class FashionDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.color_labels = []
        self.pattern_labels = []
        self.style_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.color_labels.append(self.attr.color_name_to_id[row['顏色(Color)']])
                self.pattern_labels.append(self.attr.pattern_name_to_id[row['圖案(Pattern Type)']])
                self.style_labels.append(self.attr.style_name_to_id[row['風格(Style)']])

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
        dict_data = {
            'img': img,
            'labels': {
                'color_labels': self.color_labels[idx],
                'pattern_labels': self.pattern_labels[idx],
                'style_labels': self.style_labels[idx]
            }
        }
        return dict_data