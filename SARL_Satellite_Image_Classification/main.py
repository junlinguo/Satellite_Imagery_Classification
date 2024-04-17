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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # convert the targets to the same format as inputs
        targets = targets.squeeze(1) if targets.ndim > 1 else targets
        targets = targets.long()
        # one hot targets
        one_hot_targets = F.one_hot(targets, num_classes=inputs.shape[1])


        BCE_loss = F.binary_cross_entropy_with_logits(inputs, one_hot_targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        alpha_factor = one_hot_targets * self.alpha + (1 - one_hot_targets) * (1 - self.alpha)
        F_loss = alpha_factor * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


# Log the experiment number
trial = 3

############################ model name ###############################################################################

model_name = 'resnet50' # best in data limited supervised training
# model_name = 'swin-t'
# model_name = 'dinov2_vitl14'

############################ hyper parameters setup ####################################################################
train_setup = {'img_label': '/home/guoj5/Desktop/correct_labels/Train.csv',
               'batch_size': 16,
               'epoch': 50,
               'lr': 1e-5}
val_setup = {'img_label': '/home/guoj5/Desktop/correct_labels/Valid.csv',
             'batch_size': 16,
             'output_path':'/home/guoj5/Desktop/metrics/ver2/swin-t'}

if not os.path.isdir(val_setup['output_path']):
    os.makedirs(val_setup['output_path'], exist_ok=True)

if not os.path.isdir(os.path.join(val_setup['output_path'], str(trial))):
    os.makedirs(os.path.join(val_setup['output_path'], str(trial)), exist_ok=True)

val_setup['output_path'] = os.path.join(val_setup['output_path'], str(trial))
log_minibatch = 100


# data augmentation
data_augmentation_transform = A.Compose([
        A.Flip(),
        A.RandomRotate90(),

        ## ADD
        # A.ShiftScaleRotate(p=0.5),
        # A.RandomBrightnessContrast(p=0.5)
        # ADD
        # A.Sharpen(p=0.75),
        # A.Solarize(threshold=0.05),
        # A.GaussianBlur(p=0.5)
    ])

# dataset and dataloader
train_dataset = CustomImageDataset(data_file= train_setup['img_label'],
                                   transform=data_augmentation_transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=train_setup['batch_size'],
    shuffle=True
)

## get one sample
# sample, labels = next(iter(train_loader))

val_dataset = CustomImageDataset(data_file=val_setup['img_label'], transform=data_augmentation_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_setup['batch_size'],
    shuffle=False
)

# model
##### model FC layers adjustment  #####################################################################################
if model_name == 'resnet50':
    model = timm.create_model(model_name, num_classes=2, in_chans=9, pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=1024, bias=True),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(1024, 2)
    )

if model_name == 'swin-t':
    model = SwinTransformer(num_classes=2, img_size=256, in_chans=9)
    model.head.fc = nn.Sequential(
        nn.Linear(in_features=768, out_features=256, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2)
    )
if model_name == 'dinov2_vitl14':
    from models.dinov2_1 import load_dinov2_vitl14, replicate_layer
    model = load_dinov2_vitl14()
    model.patch_embed.proj = replicate_layer(model.patch_embed.proj)

optimizer = torch.optim.Adam(model.parameters(), lr=train_setup['lr'])

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Log
logs = {'auc_score': [],
        'sensitivity': [],
        'specificity': [],
        'f1-score': []
        }

criterion = nn.CrossEntropyLoss()
# criterion = FocalLoss(alpha=0.25, gamma=2.0)

# training_help
for epoch in tqdm(range(train_setup['epoch'])):  # loop over the dataset multiple times
    print()
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

