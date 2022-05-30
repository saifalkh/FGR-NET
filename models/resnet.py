import torch.nn.functional as F

import torchvision.models as models

import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

####################################################################
class ResNet(nn.Module):        # ResNet with 2 more layers for classifier

    def __init__(self, n_channels=3, n_classes=3, bilinear=False):
        super(ResNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.resnet = models.resnet152(pretrained=False)

        # FC layers
        # self.fc1 = nn.Linear(self.resnet.fc.in_features, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        # self.fc2 = nn.Linear(1024, 768)
        self.fc2 = nn.Linear(self.resnet.fc.in_features, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.fc3 = nn.Linear(768, 256)

        # We delete the last FC layer of resnet
        self.resnet.fc = Identity()

        # FC layers
        num_ftrs = 256
        self.fcl1 = nn.Linear(num_ftrs, 128)
        self.bnl1 = nn.BatchNorm1d(128)
        self.fcl2 = nn.Linear(128, 64)
        self.bnl2 = nn.BatchNorm1d(64)

        # Classifier
        num_ftrs = 64
        self.classifier = nn.Linear(num_ftrs, n_classes)
        #self.classifier1 = nn.Linear(num_ftrs, 32)
        #self.classifier2 = nn.Linear(32, n_classes)

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


    def forward(self, input):

        # Encoder
        x0 = self.resnet(input)

        # Flatten
        x = x0.view(x0.size(0), -1)

        # FC layers
        #x = self.bn1(self.fc1(x0))
        #x = self.relu(x)
        # TODO: why are these layers being added?
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)


        # classifier
        C = self.fcl1(x)            # First additional layer
        C = self.bnl1(C)
        C = self.fcl2(C)            # Second additional layer
        C = self.bnl2(C)
        out = self.classifier(C)
        #out = self.classifier1(C)
        #out = self.classifier2(out)

        # Decoder
        x1 = self.relu(self.fc_bn4(self.fc4(x)))
        x2 = self.relu(self.fc_bn5(self.fc5(x1))).view(-1, 64, 4, 4)

        x3 = self.relu(self.bn3(self.conv1(x2)))
        x4 = self.relu(self.bn4(self.conv2(x3)))
        x5 = F.softmax(self.bn5(self.conv3(x4)), dim=1)

        logits = x5
        logits = F.interpolate(logits, size=(320, 320), mode='bilinear')

        return logits, out

####################################################################
class DeepResNet(nn.Module):   # With 5 more layers for the classifier

  def __init__(self, n_channels, n_classes, bilinear=False):
    super(DeepResNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear

    self.resnet = models.resnet152(pretrained=False)

    self.fc2 = nn.Linear(self.resnet.fc.out_features, 320)
    self.bn2 = nn.BatchNorm1d(320)
    self.fc3 = nn.Linear(320, 160)

    # FC layers
    num_ftrs = 160
    self.fcl1 = nn.Linear(num_ftrs, 96)
    self.bnl1 = nn.BatchNorm1d(96)
    self.fcl2 = nn.Linear(96, 64)
    self.bnl2 = nn.BatchNorm1d(64)

    self.fcl3 = nn.Linear(64, 32)
    self.bnl3 = nn.BatchNorm1d(32)

    self.fcl4 = nn.Linear(32, 24)
    self.bnl4 = nn.BatchNorm1d(24)

    self.fcl5 = nn.Linear(24, 16)
    self.bnl5 = nn.BatchNorm1d(16)

    # Classifier
    self.classifier = nn.Linear(16, n_classes)

    # Decoder: FC layers and transposed convolution network
    self.fc4 = nn.Linear(160, 768)
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


  def forward(self, input):

    # Encoder
    x0 = self.resnet(input)

    # Flatten
    x = x0.view(x0.size(0), -1)

    # FC layers
    #x = self.bn1(self.fc1(x0))
    #x = self.relu(x)
    x = self.fc2(x)
    #x = self.bn2(x)
    x = self.relu(x)
    x = self.fc3(x)

    C = self.fcl1(x)      # First additional layer
    #C = self.bnl1(C)
    C = self.fcl2(C)      # Second additional layer
    #C = self.bnl2(C)
    C = self.fcl3(C)      # Third additional layer
    #C = self.bnl3(C)
    C = self.fcl4(C)      # Fourth additional layer
    #C = self.bnl4(C)
    C = self.fcl5(C)      # Fifth additional layer
    #C = self.bnl5(C)


    # Classifier
    out = self.classifier(C)

    # Decoder
    # x1 = self.relu(self.fc_bn4(self.fc4(x)))
    # x2 = self.relu(self.fc_bn5(self.fc5(x1))).view(-1, 64, 4, 4)

    # x3 = self.relu(self.bn3(self.conv1(x2)))
    # x4 = self.relu(self.bn4(self.conv2(x3)))
    # #x5 = self.sigmoid(self.bn3(self.conv3(x4)))
    # x5 = F.softmax(self.bn5(self.conv3(x4)), dim=1)
    x1 = self.relu(self.fc4(x))
    x2 = self.relu(self.fc5(x1)).view(-1, 64, 4, 4)

    x3 = self.relu(self.conv1(x2))
    x4 = self.relu(self.conv2(x3))
    x5 = F.softmax(self.conv3(x4), dim=1)

    logits = x5
    logits = F.interpolate(logits, size=(320, 320), mode='bilinear')

    return logits, out
