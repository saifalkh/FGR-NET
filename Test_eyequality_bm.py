from __future__ import print_function,division,absolute_import

import os
import cv2
import csv
import shutil
import argparse
import warnings
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import seaborn as sns
from pathlib import Path
import torch, torchvision
from matplotlib import rc
from pylab import rcParams
from PIL.Image import Image
import torch.optim as optim
from sklearn import metrics
from matplotlib import pyplot
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as T
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from matplotlib.ticker import MaxNLocator
from torchvision.datasets import ImageFolder
from torchvision import datasets,models,transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# here call the model you are using
from UNetvgg import *

#===========================================================================#
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
normalization_mu = [0.485, 0.456, 0.406]
normalization_sigma= [0.229, 0.224, 0.225]
mu,sigma=(torch.tensor(p).to(device) for p in [normalization_mu,normalization_sigma])
inverse_normalize_transform = transforms.Normalize(
    mean=-mu/sigma,
    std=1/sigma)
normalize_transform = transforms.Normalize(mu,sigma)

inverse_normalize_transform_cpu = transforms.Normalize(
    mean=-mu.cpu()/sigma.cpu(),
    std=1/sigma.cpu())

#===========================================================================#
def show_confusion_matrix(confusion_matrix, class_names):
  cm = confusion_matrix.copy()
  cell_counts = cm.flatten()
  cm_row_norm = cm / cm.sum(axis=1)[:, np.newaxis]
  row_percentages = ["{0:.2f}".format(value) for value in cm_row_norm.flatten()]
  cell_labels = [f"{cnt}\n{per}" for cnt, per in zip(cell_counts, row_percentages)]
  cell_labels = np.asarray(cell_labels).reshape(cm.shape[0], cm.shape[1])
  df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)
  hmap = sns.heatmap(df_cm, annot=cell_labels, fmt="", cmap="Blues")
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
  plt.ylabel('True Sign')
  plt.xlabel('Predicted Sign')
  plt.show()

#===========================================================================#
def getConfusionMatrix(cnf_matrix):
    sum_correct = cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[2][2]
    total = cnf_matrix[0][0] + cnf_matrix[1][1] + cnf_matrix[2][2] + cnf_matrix[0][1] + cnf_matrix[1][0] + cnf_matrix[0][2] + cnf_matrix[1][2] + cnf_matrix[2][0] + cnf_matrix[2][1]
    ACC = sum_correct / total
    print("accuracy : ", ACC)

#===========================================================================#
def test_model(imagesize, model_name, model, dataloaders, n_examples, device):
  model.load_state_dict(torch.load('checkpoint/'+model_name+'/best_model_state.ckpt'))
  model = model.eval()
  class_names = ['0','1','3']#,
  df = pd.DataFrame(columns=["No","correct","predict"])
  i=1
  correct_predictions = 0
  with torch.no_grad():
    for inputs, labels in tqdm(dataloaders):
      inputs = inputs.to(device)
      labels = labels.to(device)
      
      recoimage,outputs = model(inputs)
      _, preds = torch.max(outputs, dim=1)
      correct_predictions += torch.sum(preds == labels)
      df.loc[i] =  [i,labels.data.cpu().numpy()[0],preds.data.cpu().numpy()[0]]
      i +=1

  df.to_csv("result/"+model_name+"/test.csv", sep=',',index=False)
  data = pd.read_csv("result/"+model_name+"/test.csv", sep=",") 
  test_acc = correct_predictions.double() / n_examples
  print(f'Test accuracy {test_acc}')

  y_test = data['correct']
  y_pred = data['predict']
  print(classification_report(y_test, y_pred, target_names=class_names))
  cm = confusion_matrix(y_test, y_pred)
  show_confusion_matrix(cm, class_names)
  getConfusionMatrix(cm)
  print(cm)
# ================================================================ # 
from other_models import View
def create_model(model_name,device:torch.DeviceObjType, num_classes=3):
    
    if model_name == "UNetvgg16":
      model = UNet16()
      encoder = ""
      model.name = "UNetvgg16"
    
    else:
      print("Please, input the name of model")
    
    return model.to(device),encoder

if __name__ == '__main__':

  model_name = 'UNetvgg16'
  imagesize = 480
  data_dir='/home/ircvg/saif/2021_all_work/facundo/dataset'
  #data_dir='/home/ircvg/saif/2021_all_work/facundo/original_data_Q' #  3 class
  test_dir=data_dir+'/test'
  # normalization 
  data_transforms = {
      'test': transforms.Compose([
          transforms.Resize(imagesize),
          transforms.CenterCrop(imagesize),
          # If we increase the contrast :
          # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['test']}
  
  dataloaders_test= {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=True, num_workers=2)
                for x in  ['test']}
              
  dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
  class_names = image_datasets['test'].classes


  # Get a batch of training data
  inputs, classes = next(iter(dataloaders_test['test']))

  # Make a grid from batch
  out = torchvision.utils.make_grid(inputs)
  model, encoder = create_model(model_name,device,len(class_names))
  test_model(imagesize, model_name, model,dataloaders_test['test'], dataset_sizes['test'], device)
