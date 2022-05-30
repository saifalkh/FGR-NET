from __future__ import print_function,division,absolute_import
import os
import cv2
import shutil
import pathlib
import torchvision
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from glob import glob
import torch.nn as nn
import seaborn as sns
import torch.autograd
from pathlib import Path
import torch, torchvision
from matplotlib import rc
from pylab import rcParams
from PIL.Image import Image
import torch.optim as optim
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
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from sklearn.metrics import confusion_matrix, classification_report
# Here call the model from Test_eyequality_bm(with best model)
from Test_eyequality_bm import create_model
# ================================================================ # 
def train_epoch(model,dataloaders,loss_fn,loss_MSE,optimizer,device,scheduler,n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  i = 1
  for inputs, labels in tqdm(dataloaders):
    inputs = inputs.to(device)
    labels = labels.to(device)
    recoimage, outputs = model(inputs)   
    _, preds = torch.max(outputs, dim=1)
    loss1 = loss_fn(outputs, labels)
    loss2 = loss_MSE(inputs, recoimage)     
    if i%100 == 0:
        save_image(inputs, 's/'+str(i)+'_Input.jpeg')
        save_image(recoimage, 's/'+str(i)+'_Reco.jpeg')
    i +=1
    loss = (loss1+loss2)/2 
    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  return model, correct_predictions.double() / n_examples ,np.mean(losses) # 
# ================================================================ # 
def eval_model(model, dataloaders, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for inputs, labels in tqdm(dataloaders):
      inputs = inputs.to(device)
      labels = labels.to(device)
      _, outputs = model(inputs)
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, labels)
      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())
  return correct_predictions.double() / n_examples, np.mean(losses) 

#===========================================================================#
def test_model(model, dataloaders, device, n_examples):
  model = model.eval()
  df = pd.DataFrame(columns=["No","correct","predict"])
  i=1
  correct_predictions = 0
  with torch.no_grad():
    for inputs, labels in tqdm(dataloaders):
      inputs = inputs.to(device)
      labels = labels.to(device)
      recoimage, outputs = model(inputs)
      #if i%100 == 0:
      #save_image(recoimage, 's/img_'+str(i)+'.jpeg')
      _, preds = torch.max(outputs, dim=1)
      correct_predictions += torch.sum(preds == labels)
      df.loc[i] =  [i,labels.data.cpu().numpy()[0],preds.data.cpu().numpy()[0]]
      i +=1

  df.to_csv("result/test.csv", sep=',',index=False)
  return correct_predictions.double() / n_examples
#-----------------------------------------------------------
# ================================================================ # 
def checkpoint_path(filename,model_name):
  
  checkpoint_folderpath = pathlib.Path(f'checkpoint/{model_name}')
  print(checkpoint_folderpath)
  checkpoint_folderpath.mkdir(exist_ok=True,parents=True)
  return checkpoint_folderpath/filename
# ================================================================ # 
def train_model(model, dataloaders, dataloaders_test, dataset_sizes, device, n_epochs=50):
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)#0.01 20 epoch and reduse to 0.001 after 10 epoch
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  loss_fn = nn.CrossEntropyLoss(reduction='mean').to(device)
  loss_MSE = nn.MSELoss().to(device)
  best_model_path = checkpoint_path('best_model_state.ckpt',model.name)
  # model.load_state_dict(torch.load(best_model_path))
  model.eval()
  print(model)
  
  history = defaultdict(list)
  best_accuracy = 0
  for epoch in range(0,n_epochs):
    print(f'Epoch {epoch + 1}/{n_epochs}')
    print('-' * 10)
    model, train_acc, train_loss = train_epoch(model,dataloaders['train'],loss_fn,loss_MSE,optimizer,device,scheduler,dataset_sizes['train'])
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = eval_model(model,dataloaders['val'],loss_fn,device,dataset_sizes['val'])
    print(f'validation   loss {val_loss} accuracy {val_acc}')
    test_acc = test_model(model, dataloaders_test['test'], device, dataset_sizes['test'])
    print(f'Test accuracy {test_acc}')  
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    torch.save(model.state_dict(), checkpoint_path('best_model_state_'+str(epoch)+'.ckpt',model.name))
    if test_acc > best_accuracy:
      torch.save(model.state_dict(), best_model_path)
      best_accuracy = test_acc
  print(f'Best val accuracy: {best_accuracy}')
  model.load_state_dict(torch.load(best_model_path))
  return model, history
# ================================================================ #  
def plot_training_history(history):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
  ax1.plot(history['train_loss'], label='train loss')
  ax1.plot(history['val_loss'], label='validation loss')
  ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax1.set_ylim([-0.05, 1.05])
  ax1.legend()
  ax1.set_ylabel('Loss')
  ax1.set_xlabel('Epoch')
  ax2.plot(history['train_acc'], label='train accuracy')
  ax2.plot(history['val_acc'], label='validation accuracy')
  ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
  ax2.set_ylim([-0.05, 1.05])
  ax2.legend()
  ax2.set_ylabel('Accuracy')
  ax2.set_xlabel('Epoch')
  fig.suptitle('Training history')
# ================================================================ #

if __name__ == '__main__':

  # ================================================================ # 
  # dataset path
  data_dir='/home/ircvg/saif/2021_all_work/facundo/original_data_Q' #  3 class
  train_dir=data_dir+'/train'
  valid_dir=data_dir+'/val'
  test_dir=data_dir+'/test'
  # ================================================================ # 
  # Data augmentation and normalization for training
  # Just normalization for validation
  resolution = 480
  data_transforms = {
      'train': transforms.Compose([
        transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.RandomHorizontalFlip(),
          transforms.RandomRotation(180),
          transforms.ColorJitter(brightness=0.01,contrast=0.01,hue=0.01,saturation=0.01),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'test': transforms.Compose([
          transforms.Resize(resolution),
          transforms.CenterCrop(resolution),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }
  # ================================================================ # 
  model_name = 'UNetvgg16' 

  image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train','val', 'test']}
  dataloaders= {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=2)
                for x in  ['train','val']}
  
  dataloaders_test= {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                              shuffle=False, num_workers=2,drop_last=False)
                for x in  ['test']}
              
  dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val', 'test']}
 

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # ================================================================ # 
  print("Device: ", device)
  class_names = image_datasets['train'].classes
  print("Classes: ", class_names)
  # ================================================================ # 
  base_model, encoder = create_model(model_name,num_classes=len(class_names),device=device)
  # ================================================================ # 
  base_model, history = train_model(base_model, dataloaders, dataloaders_test, dataset_sizes, device)
  # ================================================================ # 
  plot_training_history(history)
  # ================================================================ # 
