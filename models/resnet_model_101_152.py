""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

#from .unet_parts import *

import torchvision.models as models

import torch
import torch.nn as nn
#from torchvision import models
#from models import modules
import torchvision
from torchvision.utils import save_image


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

####################################################################
class ResNet152(nn.Module):   # First version of ResNet (except pretrained parameters)

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ResNet152, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.resnet = models.resnet152(pretrained=False)

        # We delete the last FC layer of resnet
        #modules = list(self.resnet.children())[:-1]
        #self.resnet = nn.Sequential(*modules)

        # FC layers
        #self.fc1 = nn.Linear(self.resnet.fc.in_features, 1024)
        #self.bn1 = nn.BatchNorm1d(1024)
        #self.fc2 = nn.Linear(1024, 768)

        self.fc2 = nn.Linear(self.resnet.fc.in_features, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.fc3 = nn.Linear(768, 256)

        # We delete the last FC layer of resnet
        self.resnet.fc = Identity()

        # Classifier
        num_ftrs = 256
        self.classifier = nn.Linear(num_ftrs, n_classes)

        # Decoder: FC layers and transposed convolution network
        self.fc4 = nn.Linear(256, 768)
        self.fc_bn4 = nn.BatchNorm1d(768)
        self.fc5 = nn.Linear(768, 64*4*4)     # 64*4*4 = 1024
        self.fc_bn5 = nn.BatchNorm1d(64*4*4)

        self.conv1 = nn.ConvTranspose2d(64, 32, 3, 2)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self. conv2 = nn.ConvTranspose2d(32, 8, 3, 2)
        self.bn4 = nn.BatchNorm2d(8)

        self.conv3 = nn.ConvTranspose2d(8, 3, 3, 2)
        self.bn5 = nn.BatchNorm2d(3)
        #self.sigmoid = nn.Sigmoid()


    def forward(self, input):

        # Encoder
        x0 = self.resnet(input)

        # Flatten
        x = x0.view(x0.size(0), -1)

        # FC layers
        #x = self.bn1(self.fc1(x0))
        #x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)

        # Classifier
        out = self.classifier(x)
        #print("out=",out.shape)

        # Decoder
        x1 = self.relu(self.fc_bn4(self.fc4(x)))
        x2 = self.relu(self.fc_bn5(self.fc5(x1))).view(-1, 64, 4, 4)

        x3 = self.relu(self.bn3(self.conv1(x2)))
        x4 = self.relu(self.bn4(self.conv2(x3)))
        #x5 = self.sigmoid(self.bn3(self.conv3(x4)))
        x5 = F.softmax(self.bn5(self.conv3(x4)), dim=1)

        logits = x5
        logits = F.interpolate(logits, size=(320, 320), mode='bilinear')
        #print("logits=",logits.shape)

        return logits, out

####################################################################
class ResNet101(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ResNet101, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.resnet = models.resnet101(pretrained=True)

        # We delete the last FC layer of resnet
        #modules = list(self.resnet.children())[:-1]
        #self.resnet = nn.Sequential(*modules)

        # FC layers
        #self.fc1 = nn.Linear(self.resnet.fc.in_features, 1024)
        #self.bn1 = nn.BatchNorm1d(1024)
        #self.fc2 = nn.Linear(1024, 768)

        self.fc2 = nn.Linear(self.resnet.fc.in_features, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.fc3 = nn.Linear(768, 256)

        # We delete the last FC layer of resnet
        self.resnet.fc = Identity()

        # Classifier
        num_ftrs = 256
        self.classifier = nn.Linear(num_ftrs, n_classes)

        # Decoder: FC layers and transposed convolution network
        self.fc4 = nn.Linear(256, 768)
        self.fc_bn4 = nn.BatchNorm1d(768)
        self.fc5 = nn.Linear(768, 64*4*4)     # 64*4*4 = 1024
        self.fc_bn5 = nn.BatchNorm1d(64*4*4)

        self.conv1 = nn.ConvTranspose2d(64, 32, 3, 2)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self. conv2 = nn.ConvTranspose2d(32, 8, 3, 2)
        self.bn4 = nn.BatchNorm2d(8)

        self.conv3 = nn.ConvTranspose2d(8, 3, 3, 2)
        self.bn5 = nn.BatchNorm2d(3)
        #self.sigmoid = nn.Sigmoid()


    def forward(self, input):

        # Encoder
        x0 = self.resnet(input)

        # Flatten
        x = x0.view(x0.size(0), -1)

        # FC layers
        #x = self.bn1(self.fc1(x0))
        #x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)

        # Classifier
        out = self.classifier(x)
        #print("out=",out.shape)

        # Decoder
        x1 = self.relu(self.fc_bn4(self.fc4(x)))
        x2 = self.relu(self.fc_bn5(self.fc5(x1))).view(-1, 64, 4, 4)

        x3 = self.relu(self.bn3(self.conv1(x2)))
        x4 = self.relu(self.bn4(self.conv2(x3)))
        #x5 = self.sigmoid(self.bn3(self.conv3(x4)))
        x5 = F.softmax(self.bn5(self.conv3(x4)), dim=1)

        logits = x5
        logits = F.interpolate(logits, size=(320, 320), mode='bilinear')
        #print("logits=",logits.shape)

        return logits, out