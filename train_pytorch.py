

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
   since = time.time()

   best_model_wts = copy.deepcopy(model.state_dict())
   best_acc = 0.0
   metrics = []
   for epoch in range(num_epochs):
       print('Epoch {}/{}'.format(epoch, num_epochs - 1))
       print('-' * 10)
       metrics_data = {'epoch': epoch, 'accuracy': None,
                   'loss': None, 'val_accuracy': None,
                   'val_loss': None, class_names[0]+'_acuracy': None,
                    class_names[1]+'_acuracy': None}
       epoch_preds = []
       epoch_labels = []
       # Each epoch has a training and validation phase
       for phase in ['train', 'validation']:
           if phase == 'train':
               model.train()  # Set model to training mode
           else:
               model.eval()   # Set model to evaluate mode

           running_loss = 0.0
           running_corrects = 0

           # Iterate over data.
           for inputs, labels in dataloaders[phase]:
               inputs = inputs.to(device)
               labels = labels.to(device)

               # zero the parameter gradients
               optimizer.zero_grad()

               # forward
               # track history if only in train
               with torch.set_grad_enabled(phase == 'train'):
                   outputs = model(inputs)
                   _, preds = torch.max(outputs, 1)
                   loss = criterion(outputs, labels)

                   # backward + optimize only if in training phase
                   if phase == 'train':
                       loss.backward()
                       optimizer.step()

               # statistics
               running_loss += loss.item() * inputs.size(0)
               running_corrects += torch.sum(preds == labels.data)
               if phase == 'validation':
                 epoch_preds.extend(preds.tolist())
                 epoch_labels.extend(labels.data.tolist())
           
           epoch_loss = running_loss / dataset_sizes[phase]
           epoch_acc = running_corrects.double() / dataset_sizes[phase]

           if phase == 'train':
               scheduler.step()
               metrics_data['accuracy'] = epoch_acc.item()
               metrics_data['loss'] = epoch_loss
           
           print('{} Loss: {:.4f} Acc: {:.4f}'.format(
               phase, epoch_loss, epoch_acc))

           # deep copy the model
           if phase == 'validation':
               if epoch_acc > best_acc:
                   best_acc = epoch_acc
                   best_model_wts = copy.deepcopy(model.state_dict())
               metrics_data['val_accuracy'] = epoch_acc.item()
               metrics_data['val_loss'] = epoch_loss
       cm = confusion_matrix(epoch_labels, epoch_preds)
       metrics_data[class_names[0]+'_acuracy'] = cm[0][0]/(cm[0][0]+cm[1][0])
       metrics_data[class_names[1]+'_acuracy'] = cm[1][1]/(cm[0][1]+cm[1][1])
       metrics.append(metrics_data)
       metrics_df = pd.DataFrame(metrics)
       metrics_df.to_csv("metrics.csv", index=False)

   time_elapsed = time.time() - since
   print('Training complete in {:.0f}m {:.0f}s'.format(
       time_elapsed // 60, time_elapsed % 60))
   print('Best val Acc: {:4f}'.format(best_acc))
   
   # load best model weights
   model.load_state_dict(best_model_wts)
   return model




data_transforms = {
   'train': transforms.Compose([
       transforms.Resize((150, 150)),
       # transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
   ]),
   'validation': transforms.Compose([
       transforms.Resize((150, 150)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
   ]),
}

data_dir = 'data/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                         data_transforms[x])
                 for x in ['train', 'validation']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10,
                                            shuffle=True, num_workers=4)
             for x in ['train', 'validation']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Classes:", class_names)

model_ft = models.vgg16(pretrained=True)
model_ft.classifier = nn.Sequential(
  nn.Linear(in_features=25088, out_features=4096, bias=True),
  nn.ReLU(inplace=True),
  nn.Dropout(p=0.5, inplace=False),
  nn.Linear(in_features=4096, out_features=4096, bias=True),
  nn.ReLU(inplace=True),
  nn.Dropout(p=0.5, inplace=False),
  nn.Linear(in_features=4096, out_features=2, bias=True)
)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                      num_epochs=10)

torch.save(model_ft,'model.h5')
