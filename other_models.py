import torch.nn.functional as F

from unet_parts import *

import torchvision.models as models

import torch
import torch.nn as nn
#from torchvision import models
#from models import modules
import torchvision
from torchvision.utils import save_image

# need a special version of unpool to export to onnx
from models.maxunpool2d import MaxUnpool2d

class View(nn.Module):
    def __init__(self, shape):
      super().__init__()
      self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)
        # Linear layer
        self.classifier = nn.Linear(1024, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        out = F.relu(x5, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return logits, out


##########################################################################

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class UNet1(nn.Module):
    def __init__(self, num_classes, input_channels=3, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output

class NestedUNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output

##########################################################################
class VGGBigBlock(nn.Module):
    def __init__(self, in_channels, middle_channels1, middle_channels2, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels1, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels1)
        self.conv2 = nn.Conv2d(middle_channels1, middle_channels2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(middle_channels2)
        self.conv3 = nn.Conv2d(middle_channels2, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class SegNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SegNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)

        # Linear layer
        self.classifier = nn.Linear(512, n_classes)

        nb_filter = [64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        self.unpool = MaxUnpool2d(2,2)

        # The encoder is the same
        self.conv0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0])
        self.conv1 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2 = VGGBigBlock(nb_filter[1], nb_filter[2], nb_filter[2], nb_filter[2])
        self.conv3 = VGGBigBlock(nb_filter[2], nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv4 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[3])

        # Decoder
        self.conv5 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv6 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[2])
        self.conv7 = VGGBigBlock(nb_filter[2], nb_filter[2], nb_filter[2], nb_filter[1])
        self.conv8 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[0])

        self.relu9 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1)
        self.bn9 = nn.BatchNorm2d(nb_filter[0])
        self.conv10 = nn.Conv2d(nb_filter[0], n_classes, 3, padding=1)


    def forward(self, input):

        # /!\ This version is not pre-trained with vgg weights !

        # Encoder
        x0 = self.conv0(input)
        x1_pool, max_indices1 = self.pool(x0)
        x1 = self.conv1(x1_pool)

        x2_pool, max_indices2 = self.pool(x1)
        x2 = self.conv2(x2_pool)

        x3_pool, max_indices3 = self.pool(x2)
        x3 = self.conv3(x3_pool)

        x4_pool, max_indices4 = self.pool(x3)
        x4 = self.conv4(x4_pool)

        x5, max_indices5 = self.pool(x4)

        # Decoder
        x6 = self.conv5(self.unpool(x5, max_indices5))
        x7 = self.conv6(self.unpool(x6, max_indices4))
        x8 = self.conv7(self.unpool(x7, max_indices3))
        x9 = self.conv8(self.unpool(x8, max_indices2))

        x10 = self.relu9(self.bn9(self.conv9(self.unpool(x9,max_indices1))))
        x10 = self.conv10(x10)

        logits = F.softmax(x10, dim=1)

        # Classifier
        out = F.relu(x5, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return logits, out

####################################################################
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SegNetWithTorchModel(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SegNetWithTorchModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.outc = OutConv(64, n_classes)
        self.unpool = nn.MaxUnpool2d(2,2)

        # Linear layer
        self.classifier1 = nn.Linear(512, 256)
        self.classifier2 = nn.Linear(256, 128)
        self.classifier3 = nn.Linear(128, n_classes)

        nb_filter = [64, 128, 256, 512]

        self.inc = DoubleConv(n_channels, 64)

        # Torch model of VGG16 :
        self.vgg16 = models.vgg16_bn(pretrained=True)

        # We have to remove the classifier and the avgpool :
        self.vgg16.avgpool = Identity()
        self.vgg16.classifier = Identity()

        # Decoder
        self.conv5 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[3])
        self.conv6 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[2])
        self.conv7 = VGGBigBlock(nb_filter[2], nb_filter[2], nb_filter[2], nb_filter[1])
        self.conv8 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[0])

        self.relu9 = nn.ReLU(inplace=True)
        self.conv9 = nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1)
        self.bn9 = nn.BatchNorm2d(nb_filter[0])
        self.conv10 = nn.Conv2d(nb_filter[0], n_channels, 3, padding=1)

        self.max1 =  nn.Sequential(*list(self.vgg16.features.children())[:43],
                                   nn.MaxPool2d(2, 2, return_indices=True))
        self.max2 =  nn.Sequential(*list(self.vgg16.features.children())[:33],
                                   nn.MaxPool2d(2, 2, return_indices=True))
        self.max3 =  nn.Sequential(*list(self.vgg16.features.children())[:23],
                                   nn.MaxPool2d(2, 2, return_indices=True))
        self.max4 =  nn.Sequential(*list(self.vgg16.features.children())[:13],
                                   nn.MaxPool2d(2, 2, return_indices=True))
        self.max5 =  nn.Sequential(*list(self.vgg16.features.children())[:6],
                                   nn.MaxPool2d(2, 2, return_indices=True))


    def forward(self, input):

        # Encoder
        x1 = self.vgg16(input)
        x1 = x1.reshape(-1,512,15,15) # 10=10 =>320

        # Max indices
        y,max1 = self.max1(input)
        y,max2 = self.max2(input)
        y,max3 = self.max3(input)
        y,max4 = self.max4(input)
        y,max5 = self.max5(input)

        # Decoder
        x2 = self.conv5(self.unpool(x1, max1))
        x3 = self.conv6(self.unpool(x2, max2))
        x4 = self.conv7(self.unpool(x3, max3))
        x5 = self.conv8(self.unpool(x4, max4))
        x5 = self.relu9(self.bn9(self.conv9(self.unpool(x5, max5))))
        x10 = self.conv10(x5)

        logits = F.softmax(x10, dim=1)
        # logits = F.interpolate(logits, size=(320, 320), mode='bilinear')

        # Classifier
        out = F.relu(x1, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier1(out)
        out = self.classifier2(out)
        out = self.classifier3(out)

        return logits, out


# class SegNetWithTorchModel(nn.Module):

#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(SegNetWithTorchModel, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = DoubleConv(n_channels, 64)
#         self.outc = OutConv(64, n_classes)
#         self.unpool = MaxUnpool2d(2,2)

#         # Linear layer
#         self.classifier1 = nn.Linear(512, 256)
#         self.classifier2 = nn.Linear(256, 128)
#         self.classifier3 = nn.Linear(128, n_classes)

#         nb_filter = [64, 128, 256, 512]

#         self.inc = DoubleConv(n_channels, 64)

#         # Torch model of VGG16 :
#         self.vgg16 = models.vgg16_bn(pretrained=True)

#         # We have to remove the classifier and the avgpool :
#         self.vgg16.avgpool = Identity()
#         self.vgg16.classifier = Identity()

        
#         # Decoder
#         self.conv5 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[3])
#         self.conv6 = VGGBigBlock(nb_filter[3], nb_filter[3], nb_filter[3], nb_filter[2])
#         self.conv7 = VGGBigBlock(nb_filter[2], nb_filter[2], nb_filter[2], nb_filter[1])
#         self.conv8 = VGGBlock(nb_filter[1], nb_filter[1], nb_filter[0])

#         self.relu9 = nn.ReLU(inplace=True)
#         self.conv9 = nn.Conv2d(nb_filter[0], nb_filter[0], 3, padding=1)
#         self.bn9 = nn.BatchNorm2d(nb_filter[0])
#         self.conv10 = nn.Conv2d(nb_filter[0], n_channels, 3, padding=1)

#         self.max1 =  nn.Sequential(*list(self.vgg16.features.children())[:43],
#                                    nn.MaxPool2d(2, 2, return_indices=True))
#         self.max2 =  nn.Sequential(*list(self.vgg16.features.children())[:33],
#                                    nn.MaxPool2d(2, 2, return_indices=True))
#         self.max3 =  nn.Sequential(*list(self.vgg16.features.children())[:23],
#                                    nn.MaxPool2d(2, 2, return_indices=True))
#         self.max4 =  nn.Sequential(*list(self.vgg16.features.children())[:13],
#                                    nn.MaxPool2d(2, 2, return_indices=True))
#         self.max5 =  nn.Sequential(*list(self.vgg16.features.children())[:6],
#                                    nn.MaxPool2d(2, 2, return_indices=True))


#     def forward(self, input):

#         # Encoder
#         x1 = self.vgg16(input)
#         x1 = x1.reshape(-1,512,10,10)

#         # Max indices
#         y,max1 = self.max1(input)
#         y,max2 = self.max2(input)
#         y,max3 = self.max3(input)
#         y,max4 = self.max4(input)
#         y,max5 = self.max5(input)

#         # Decoder
#         x2 = self.conv5(self.unpool(x1, max1))
#         x3 = self.conv6(self.unpool(x2, max2))
#         x4 = self.conv7(self.unpool(x3, max3))
#         x5 = self.conv8(self.unpool(x4, max4))
#         x5 = self.relu9(self.bn9(self.conv9(self.unpool(x5, max5))))
#         x10 = self.conv10(x5)

#         logits = F.softmax(x10, dim=1)
#         logits = F.interpolate(logits, size=(320, 320), mode='bilinear')

#         # Classifier
#         out = F.relu(x1, inplace=False)
#         out = F.adaptive_avg_pool2d(out, (1, 1))
#         out = torch.flatten(out, 1)
#         out = self.classifier1(out)
#         out = self.classifier2(out)
#         out = self.classifier3(out)

#         return logits, out

####################################################################
class DenseNet(nn.Module):

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DenseNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # DenseNet
        self.densenet = models.densenet161(pretrained=True)

        # For the classification at the end of the encoder :
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, n_classes)

        # For the reconstruction of the image :
        self.densenet2 = models.densenet161(pretrained=True)

        # FC layers
        self.fc1 = nn.Linear(self.densenet2.classifier.in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 768)
        self.bn2 = nn.BatchNorm1d(768)
        self.fc3 = nn.Linear(768, 256)

        # We delete the classifier of densenet2
        self.densenet2.classifier = Identity()

        # Decoder: The same as ResNet
        self.fc4 = nn.Linear(256, 768)
        self.fc_bn4 = nn.BatchNorm1d(768)
        self.fc5 = nn.Linear(768, 64*4*4)
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
        out = self.densenet(input)
        x0 = self.densenet2(input)

        # Flatten
        x = x0.view(x0.size(0), -1)

        # FC layers
        x = self.bn1(self.fc1(x0))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.fc3(x)

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
class DenseNetEncoder(nn.Module):
  def __init__(self):
    super(DenseNetEncoder, self).__init__()
    self.densenet = models.densenet161(pretrained = True)

  def forward(self, input):
    # We change the input into a vector
    features = [input]
    # We add the items of the model (starting from the features) to the vectorized input
    for k, v in self.densenet.features._modules.items(): features.append( v(features[-1]) )
    return features

class UpSample(nn.Sequential):
  def __init__(self, skip_input, output_features):
    super(UpSample, self).__init__()
    self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
    self.leakyreluA = nn.LeakyReLU(0.2)
    self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
    self.leakyreluB = nn.LeakyReLU(0.2)

  def forward(self, x, concat_with):
    # "interpolate" in order to match with a correct size
    up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
    return self.leakyreluB(self.convB(self.leakyreluA(self.convA(torch.cat([up_x, concat_with], dim=1)))))
    # We can replace the previous line by the following line (no leakyreluA) :
    # return self.leakyreluB(self.convB(self.convA(torch.cat([up_x, concat_with], dim=1))))

class DenseNetDecoder(nn.Module):
  # 2208: the size of the in_features of the Linear classifier at the end of DenseNet161
  def __init__(self, num_features = 2208, decoder_width = 0.5):
    super(DenseNetDecoder, self).__init__()
    self.num_features = num_features
    self.decoder_width = decoder_width
    self.features = int(self.num_features * self.decoder_width)
    self.conv2 = nn.Conv2d(self.num_features, self.features, kernel_size=1, stride=1, padding=1)

    # U-Net decoder :
    nb_filter = [96, 96, 192, 384]
    self.conv3_1 = VGGBlock(self.features//1 + nb_filter[3], self.features//2, self.features//2)
    self.conv2_2 = VGGBlock(self.features//2 + nb_filter[2], self.features//4, self.features//4)
    self.conv1_3 = VGGBlock(self.features//4 + nb_filter[1], self.features//8, self.features//8)
    self.conv0_4 = VGGBlock(self.features//8 + nb_filter[0], self.features//16, self.features//16)
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

  def forward(self, features):
    x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[11]

    x_d0 = self.conv2(x_block4)
    up_x_block = self.up(x_block3)
    x_d0 = F.interpolate(x_d0, size = [up_x_block.size(2), up_x_block.size(3)], mode='bilinear', align_corners=True)
    x_d1 = self.conv3_1(torch.cat([x_d0, self.up(x_block3)], 1))

    up_x_block = self.up(x_block2)
    x_d1 = F.interpolate(x_d1, size = [up_x_block.size(2), up_x_block.size(3)], mode='bilinear', align_corners=True)
    x_d2 = self.conv2_2(torch.cat([x_d1, self.up(x_block2)], 1))

    up_x_block = self.up(x_block1)
    x_d2 = F.interpolate(x_d2, size = [up_x_block.size(2), up_x_block.size(3)], mode='bilinear', align_corners=True)
    x_d3 = self.conv1_3(torch.cat([x_d2, self.up(x_block1)], 1))

    up_x_block = self.up(x_block0)
    x_d3 = F.interpolate(x_d3, size = [up_x_block.size(2), up_x_block.size(3)], mode='bilinear', align_corners=True)
    x_d4 = self.conv0_4(torch.cat([x_d3, self.up(x_block0)], 1))

    return x_d4

class UNetWithDenseNetEncoder(nn.Module):
  def __init__(self, n_channels, n_classes, bilinear=False, align_corners=True):
    super(UNetWithDenseNetEncoder, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear

    self.encoder = DenseNetEncoder()
    # self.decoder = DenseNetDecoder()
    self.final = nn.Conv2d(69, 3, kernel_size=1)

    # For the classification at the end of the encoder :
    self.densenet = models.densenet121(pretrained=True)
    num_ftrs = self.densenet.classifier.in_features
    # self.densenet.classifier = nn.Linear(num_ftrs, num_ftrs/2)
    # self.densenet.classifier = nn.Linear(num_ftrs/2, n_classes/4)
    self.densenet.classifier = nn.Linear(num_ftrs, n_classes)
        

  
  def forward(self, input):
    # Classifier
    #out = self.densenet(input)

    # Decoder
    x = self.encoder(input)# self.decoder(self.encoder(input))
    x = self.final(x)
    #logits = x
    
    return x # logits, out

####################################################################
class ResNetEncoder(nn.Module):
  def __init__(self):
    super(ResNetEncoder, self).__init__()
    self.resnet = models.resnet50(pretrained=True)
    # self.resnet = models.resnet50(pretrained=True)

  def forward(self, input):
    # We change the input into a vector
    features = [input]
    # We add the items of the model (starting from the features) to the vectorized input
    for v in nn.Sequential(*list(self.resnet.children())[:-1]): features.append(v(features[-1]))
    return features

class ResNetDecoder(nn.Module):
  # 2048: the size of the features at the end of ResNet152
  def __init__(self, num_features = 2048, decoder_width = 0.5):
    super(ResNetDecoder, self).__init__()
    self.num_features = num_features
    self.decoder_width = decoder_width
    self.features = int(self.num_features * self.decoder_width)
    self.conv2 = nn.Conv2d(self.num_features, self.features, kernel_size=1, stride=1, padding=1)

    # U-Net decoder :
    # We adapt nb_filter to the size of the features of ResNet152
    nb_filter = [64, 64, 64, 512]
    self.conv3_1 = VGGBlock(self.features//1 + nb_filter[3], self.features//2, self.features//2)
    self.conv2_2 = VGGBlock(self.features//2 + nb_filter[2], self.features//4, self.features//4)
    self.conv1_3 = VGGBlock(self.features//4 + nb_filter[1], self.features//8, self.features//8)
    self.conv0_4 = VGGBlock(self.features//8 + nb_filter[0], self.features//16, self.features//16)
    self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

  def forward(self, features):
    x_block0, x_block1, x_block2, x_block3, x_block4 = features[2], features[3], features[4], features[6], features[8]

    x_d0 = self.conv2(x_block4)
    up_x_block = self.up(x_block3)
    x_d0 = F.interpolate(x_d0, size = [up_x_block.size(2), up_x_block.size(3)], mode='bilinear', align_corners=True)
    x_d1 = self.conv3_1(torch.cat([x_d0, self.up(x_block3)], 1))

    up_x_block = self.up(x_block2)
    x_d1 = F.interpolate(x_d1, size = [up_x_block.size(2), up_x_block.size(3)], mode='bilinear', align_corners=True)
    x_d2 = self.conv2_2(torch.cat([x_d1, self.up(x_block2)], 1))

    up_x_block = self.up(x_block1)
    x_d2 = F.interpolate(x_d2, size = [up_x_block.size(2), up_x_block.size(3)], mode='bilinear', align_corners=True)
    x_d3 = self.conv1_3(torch.cat([x_d2, self.up(x_block1)], 1))

    up_x_block = self.up(x_block0)
    x_d3 = F.interpolate(x_d3, size = [up_x_block.size(2), up_x_block.size(3)], mode='bilinear', align_corners=True)
    x_d4 = self.conv0_4(torch.cat([x_d3, self.up(x_block0)], 1))

    return x_d4

class UNetWithResNetEncoder(nn.Module):

  def __init__(self, n_channels, n_classes, bilinear=False, align_corners=True):
    super(UNetWithResNetEncoder, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear

    self.encoder = ResNetEncoder()
    self.decoder = ResNetDecoder()
    #self.final = nn.Conv2d(64, n_classes, kernel_size=1)
    self.final = nn.Conv2d(64, 3, kernel_size=1)

    # For the classification at the end of the encoder :
    self.resnet = models.resnet152(pretrained=True)
    num_ftrs = self.resnet.fc.in_features
    self.resnet.fc = nn.Linear(num_ftrs, n_classes)



    # Linear layer
    self.classifier1 = nn.Linear(2048, 1024)
    self.classifier2 = nn.Linear(1024, 512)
    self.classifier3 = nn.Linear(512, 256)
    self.classifier4 = nn.Linear(256, n_classes)

    # For the classification at the end of the encoder :
    # self.resnet = models.resnet50(pretrained=True)
    # num_ftrs = self.resnet.fc.in_features
    # self.resnet.fc = nn.Linear(num_ftrs, n_classes)


  def forward(self, input):
    # Classifier
    x = self.encoder(input)

    # Classifier
    out = F.relu(x[9], inplace=False)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier1(out)
    out = self.classifier2(out)
    out = self.classifier3(out)
    out = self.classifier4(out)

    # out = self.resnet(input)

    # # Decoder
    # x = self.decoder(x)
    # x = self.final(x)
    # logits = x
    
    return out # logits, out

####################################################################
class MobileNet(nn.Module):

  def __init__(self, n_channels, n_classes, bilinear=False):
    super(MobileNet, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes
    self.bilinear = bilinear

    self.mobilenet = models.mobilenet.mobilenet_v2(pretrained=True)
    # mobilenet_v3_large = models.mobilenet_v3_large()
    # mobilenet_v3_small = models.mobilenet_v3_small()

    self.fc2 = nn.Linear(self.mobilenet.classifier[1].out_features, 320)
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
    #self.sigmoid = nn.Sigmoid()


  def forward(self, input):

    # Encoder
    x0 = self.mobilenet(input)

    # Flatten
    x = x0.view(x0.size(0), -1)

    # FC layers
    #x = self.bn1(self.fc1(x0))
    #x = self.relu(x)
    x = self.fc2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.fc3(x)

    C = self.fcl1(x)
    C = self.bnl1(C)
    C = self.fcl2(C)
    C = self.bnl2(C)
    C = self.fcl3(C)
    C = self.bnl3(C)
    C = self.fcl4(C)
    C = self.bnl4(C)
    C = self.fcl5(C)
    C = self.bnl5(C)


    # Classifier
    out = self.classifier(C)

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