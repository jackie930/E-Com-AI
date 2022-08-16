import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class MultiOutputModel(nn.Module):
    def __init__(self, feature_dict, model_name):
        super().__init__()
        # todo: add model support backbones
        if model_name == 'mobilenet_v2':
            # model_ft = models.inception_v3(pretrained=use_pretrained)
            self.base_model = models.mobilenet_v2(pretrained=True).features  # take the model without classifier
            last_channel = models.mobilenet_v2().last_channel  # size of the layer before the classifier

        elif model_name == 'resnet':
            x = models.resnet18(pretrained=True)
            list(x.modules())  # to inspect the modules of your model
            self.base_model = nn.Sequential(*list(x.children())[:-2])  # strips off last linear layer
            last_channel = x.fc.in_features

        '''
        elif model_name == 'alexnet':
            x = models.alexnet(pretrained=True)
            list(x.modules())  # to inspect the modules of your model
            self.base_model = nn.Sequential(*list(x.children())[:-2])  # strips off last linear layer
            last_channel = x.classifier[6].in_features

        elif model_name == 'vgg':
            x = models.vgg11_bn(pretrained=True)
            list(x.modules())  # to inspect the modules of your model
            self.base_model = nn.Sequential(*list(x.children())[:-2])  # strips off last linear layer
            last_channel = x.classifier[6].in_features

        elif model_name == 'inception':
            x = models.inception_v3(pretrained=True)
            list(x.modules())  # to inspect the modules of your model
            self.base_model = nn.Sequential(*list(x.children())[:-2])  # strips off last linear layer
            last_channel = x.fc.in_features
        '''

        # the input for the classifier should be two-dimensional, but we will have
        # [<batch_size>, <channels>, <width>, <height>]
        # so, let's do the spatial averaging: reduce <width> and <height> to 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # create separate classifiers for our outputs
        self.key_ls = list(feature_dict.keys())

        print("<<< self.key_ls: ", self.key_ls)
        self.class_len_ls = list(feature_dict.values())
        print("<<< self.class_len_ls: ", self.class_len_ls)

        self.tasks = nn.ModuleList()
        for i in range(len(self.key_ls)):
            self.tasks.add_module(
                self.key_ls[i], nn.Sequential(nn.Dropout(p=0.2),
                                              nn.Linear(in_features=last_channel, out_features=self.class_len_ls[i])))


    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, start_dim=1)

        return_dict = {}
        #return_dict = {task: F.interpolate(self.decoders[task](shared_representation), out_size, mode='bilinear') for task in self.tasks}
        #print ("<<<model: ", self)
        #print ("<<< self.tasks", self.tasks)
        i = 0
        for task in self.tasks:
            #print("<<< key: ", i)
            return_dict[self.key_ls[i]] = task(x)
            i = i+1

        #print ("<<< return dict: ", return_dict)

        return return_dict

    def get_loss(self, net_output, ground_truth):
        loss_ls = []
        for i in self.key_ls:
            g_label = str(i)+'_labels'
            loss_ls.append(F.cross_entropy(net_output[i], ground_truth[g_label]))

        loss = sum(loss_ls)

        return_dict = {}
        for i in range(len(self.key_ls)):
            return_dict[self.key_ls[i]] = loss_ls[i]

        return loss, return_dict