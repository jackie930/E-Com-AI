import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiOutputModel(nn.Module):
    def __init__(self, feature_dict):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before the classifier
        # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size>, <channels>, <width>, <height>]
        # so, let's do the spatial averaging: reduce <width> and <height> to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # create separate classifiers for our outputs
        self.key_ls = list(feature_dict.keys())
        self.class_len_ls = list(feature_dict.values())

        for i in range(len(self.key_ls)):
            self.key_ls[i] = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=last_channel, out_features=self.class_len_ls[i])
            )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, start_dim=1)

        return_dict = {}
        for i in self.key_ls:
            return_dict[i] = self.i(x)

        return return_dict

    def get_loss(self, net_output, ground_truth):
        loss_ls = []
        for i in self.key_ls:
            loss_ls.append(F.cross_entropy(net_output[i], ground_truth[i]))

        loss = sum(loss_ls)

        return_dict = {}
        for i in range(len(self.key_ls)):
            return_dict[self.key_ls[i]] = loss_ls[i]

        return loss, return_dict