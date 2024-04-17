"""
Evaluate the model performance on the negative pool

return: the wrong predicted image, (PNG)
"""

import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader
from Custom_dataset import CustomImageDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

val_setup = {'img_label': '/home/guoj5/Desktop/Test_neg.csv',
             'batch_size': 16,
             'output_path':'/home/guoj5/Desktop/metrics/test'}

## wrong prediction in the Negative pool (Hard negative mining)
Test_neg = pd.read_csv(os.path.join(val_setup['output_path'], 'Test_negative.csv'))
Hard_Negative = Test_neg[Test_neg['label'] != (Test_neg['predict'])]['img_path'].tolist() # False positive

if not os.path.isdir(os.path.join(val_setup['output_path'], 'hard_negative')):
    os.mkdir(os.path.join(val_setup['output_path'], 'hard_negative'))

# read image and save as true color format
for i in tqdm(range(len(Hard_Negative))):
    img_path = Hard_Negative[i]
    img = np.load(img_path).transpose(1, 2 , 0)  #[h, w, c]

    RGB_img = img[:, :, (4, 2, 1)]  # rgb format

    # SAVE
    im = Image.fromarray((RGB_img*255).astype(np.uint8))
    im.save(os.path.join(val_setup['output_path'], 'hard_negative', str(i)+'.png'))

pd.DataFrame(Hard_Negative, columns=['img_path']).to_csv(os.path.join(val_setup['output_path'], 'hard_negative.csv'))
    ## without axis
    # plt.imshow(RGB_img)
    # plt.axis('off')
    # plt.savefig(os.path.join(val_setup['output_path'], 'hard_negative', str(i)+'.png'), format='png', bbox_inches='tight')
    # plt.close()