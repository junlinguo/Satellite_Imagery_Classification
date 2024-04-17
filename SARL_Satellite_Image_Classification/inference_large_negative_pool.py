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
from PIL import Image

# dataset and loader
val_setup = {'img_label': '/home/guoj5/Desktop/Test_neg.csv',
             'batch_size': 16,
             'output_path':'/home/guoj5/Desktop/metrics/test'}

data_augmentation_transform = A.Compose([
        A.Flip(),
        A.RandomRotate90(),

    ])
val_dataset = CustomImageDataset(data_file=val_setup['img_label'], transform=data_augmentation_transform)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_setup['batch_size'],
    shuffle=False
)

# model
ckpt = '/home/guoj5/Desktop/metrics/res50/model/model_epoch29.pt'
model_name = 'resnet50'
if model_name == 'resnet50':
    model = timm.create_model(model_name, num_classes=2, in_chans=9, pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=1024, bias=True),
        nn.ReLU(),
        # nn.Dropout(0.5),
        nn.Linear(1024, 2)
    )

checkpoint = torch.load(ckpt)
model.load_state_dict(checkpoint['model_state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Validation phase
model.eval()
running_corrects = 0
total = 0
all_labels = []
all_preds = []
with torch.no_grad():
    for inputs, labels in tqdm(val_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        ##### add by junlin ############
        outputs = torch.softmax(outputs, 1)
        _, preds  = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().detach().numpy())

# performance log (Remaining Negative samples)
val_accuracy = running_corrects.double() / total
print(f"\n Accuracy : {val_accuracy:.4f}")

# auc_score = roc_auc_score(all_labels, all_preds)
# print(f" auc_score : {auc_score}\n")
#
# tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
# specificity = tn / (tn + fp)
# sensitivity = tp / (tp + fn)
# f1 = f1_score(all_labels, all_preds)
# ConfusionMatrixDisplay.from_predictions(y_true=all_labels, y_pred=all_preds)
# print(f'sensitivity {sensitivity:.4f}, specificity {specificity:.4f}, f1-score {f1:.4f}')

# plt.title(f'{model_name} test conf_matrix')
# plt.savefig(os.path.join(val_setup['output_path'], f'{model_name}_test_confMatrix.png'))
# plt.show()
# plt.close()

Test_neg = pd.read_csv(val_setup['img_label'])
print(f'length of TEST NEGATIVE, {len(Test_neg)}')
Test_neg['predict'] = all_preds
print('done')
Test_neg.to_csv(os.path.join(val_setup['output_path'], 'Test_negative.csv'), index=False)




