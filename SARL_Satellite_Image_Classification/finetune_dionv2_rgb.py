"""
Train script for fine-tuning part of layers of the network model/structure
"""

import pandas as pd
import torch
from torch import nn
from torchvision import transforms, datasets
import timm
from timm.data import create_transform
from torch.utils.data import DataLoader
from Custom_dataset import CustomImageDataset
import albumentations as A
from timm.models.swin_transformer import SwinTransformer
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import numpy as np

import torch.nn.functional as F

trial = 3
## At local EXXACT machine
train_setup = {'img_label': '/home/guoj5/Desktop/correct_labels/Train.csv',
               'batch_size': 16,
               'epoch': 50,
               'lr': 1e-5}
val_setup = {'img_label': '/home/guoj5/Desktop/correct_labels/Valid.csv',
             'batch_size': 16,
             'output_path':'/home/guoj5/Desktop/metrics/ver2/dinov2'}
if not os.path.isdir(val_setup['output_path']):
    os.makedirs(val_setup['output_path'],  exist_ok=True)

if not os.path.isdir(os.path.join(val_setup['output_path'], str(trial))):
    os.makedirs(os.path.join(val_setup['output_path'], str(trial)), exist_ok=True)

val_setup['output_path'] = os.path.join(val_setup['output_path'], str(trial))

log_minibatch = 100
model_name = 'dinov2_vitl14'

# data augmentation
data_augmentation_transform = A.Compose([
        # resize to be product of 14
        A.Resize(252, 252),
        A.Flip(),
        A.RandomRotate90(),
    ])

# dataset and dataloader
train_dataset = CustomImageDataset(data_file= train_setup['img_label'],
                                   transform=data_augmentation_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=train_setup['batch_size'],
    shuffle=True
)

val_dataset = CustomImageDataset(data_file=val_setup['img_label'], transform=data_augmentation_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_setup['batch_size'],
    shuffle=False
)

if model_name == 'dinov2_vitl14':
    from models.dinov2_1 import load_dinov2_vitl14, replicate_layer, DinoVisionTransformerClassifier
    model = load_dinov2_vitl14()
    model.patch_embed.proj = replicate_layer(model.patch_embed.proj)
    model = DinoVisionTransformerClassifier(model)

# freeze and unfreeze
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the first and last layer
for param in model.transformer.patch_embed.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

# freeze optimizer
optimizer = torch.optim.Adam(
    [
        {'params': model.transformer.patch_embed.parameters()},
        {'params': model.classifier.parameters()}
    ], lr=train_setup['lr'])

# optimizer = torch.optim.Adam(model.parameters(), lr=train_setup['lr'])

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Log
logs = {'auc_score': [],
        'sensitivity': [],
        'specificity': [],
        'f1-score': []
        }

# loss
criterion = nn.CrossEntropyLoss()

# training_help
for epoch in tqdm(range(train_setup['epoch'])):  # loop over the dataset multiple times
    model.train()
    running_loss = 0.0

    ## Train Acc
    train_pred = []
    train_label = []
    running_corrects = 0
    total = 0

    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % log_minibatch == log_minibatch - 1:    # print every 200 (log_mini-batches)
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / log_minibatch))
            running_loss = 0.0

        ## add train log
        outputs = torch.softmax(outputs, 1)
        _, preds = torch.max(outputs, 1)
        train_label.extend(labels.cpu().numpy())
        train_pred.extend(preds.cpu().detach().numpy())
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

    ## training accuracy
    train_accuracy = running_corrects.double() / total
    print(f"\nEpoch {epoch + 1} Train Accuracy : {train_accuracy:.4f}")

    # Validation phase
    model.eval()
    running_corrects = 0
    total = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            ##### add by junlin ############
            outputs = torch.softmax(outputs, 1)
            _, preds  = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().detach().numpy())

    val_accuracy = running_corrects.double() / total
    print(f"\nEpoch {epoch + 1} Validation Accuracy : {val_accuracy:.4f}")

    auc_score = roc_auc_score(all_labels, all_preds)
    print(f"Epoch {epoch + 1} Validation auc_score : {auc_score}\n")

    # conf_matrix = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    f1 = f1_score(all_labels, all_preds)
    ConfusionMatrixDisplay.from_predictions(y_true=all_labels, y_pred=all_preds)
    plt.title(f'Epoch {epoch + 1} conf_matrix')

    plt.savefig(os.path.join(val_setup['output_path'], f'Epoch_{epoch + 1}_confMatrix.png'))
    # plt.show()
    plt.close()

    print(f"Epoch {epoch + 1} Validation Sensitivity : {sensitivity}\n")
    print(f"Epoch {epoch + 1} Validation Specificity : {specificity}\n")

    # save metrics
    logs['auc_score'].append(round(auc_score, 4))
    logs['sensitivity'].append(round(sensitivity, 4))
    logs['specificity'].append(round(specificity,4))
    logs['f1-score'].append(round(f1, 4))

    if not os.path.isdir(os.path.join(val_setup['output_path'], 'model')):
        os.mkdir(os.path.join(val_setup['output_path'], 'model'))

    torch.save({'model_state_dict': model.state_dict(),
                    'loss': loss.item(),
                    'cum_metrics': logs
                }, os.path.join(val_setup['output_path'], 'model', f"model_epoch{epoch + 1}.pt"))

pd.DataFrame(logs, index=np.arange(1, train_setup['epoch'] + 1)).to_csv(os.path.join(val_setup['output_path'], 'logs.csv'), index_label='epoch')

print('Finished Training')
