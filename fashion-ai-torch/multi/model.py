import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiOutputModel(nn.Module):
    def __init__(self, n_color_classes, n_gender_classes, n_article_classes):
        super().__init__()
        self.base_model = models.mobilenet_v2().features  # take the model without classifier
        last_channel = models.mobilenet_v2().last_channel  # size of the layer before the classifier
        # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size>, <channels>, <width>, <height>]
        # so, let's do the spatial averaging: reduce <width> and <height> to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # create separate classifiers for our outputs
        self.color = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_color_classes)
        )

        self.gender = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        )

        self.article = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=last_channel, out_features=n_article_classes)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, start_dim=1)

        return {
            'color': self.color(x),
            'pattern': self.gender(x),
            'style': self.article(x)
        }

    def get_loss(self, net_output, ground_truth):
        color_loss = F.cross_entropy(net_output['color'], ground_truth['color_labels'])
        gender_loss = F.cross_entropy(net_output['pattern'], ground_truth['pattern_labels'])
        article_loss = F.cross_entropy(net_output['style'], ground_truth['style_labels'])
        loss = color_loss + gender_loss + article_loss
        return loss, {'color': color_loss, 'pattern': gender_loss, 'style': article_loss}