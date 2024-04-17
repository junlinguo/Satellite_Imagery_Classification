import os
import glob
import pandas as pd
import random
import numpy as np

original_split = False
if original_split:
    all_positive_filtered = pd.read_csv('/home/guoj5/Desktop/all_positive_filtered.csv')  # len(): 1634
    all_negative_filtered = pd.read_csv('/home/guoj5/Desktop/all_negative_filtered.csv')  # len(): 79920

    # down sample
    train_pos = all_positive_filtered.iloc[:1500]
    valid_pos = all_positive_filtered.iloc[1500:]

    # shuffle negative samples
    all_negative_filtered = all_negative_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
    train_neg = all_negative_filtered.iloc[:1500]
    valid_neg = all_negative_filtered.iloc[1500:1500 + 10 * len(valid_pos)]

    # concat pos and neg for dataset
    Train = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
    Valid = pd.concat([valid_pos, valid_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

    Train.to_csv('/home/guoj5/Desktop/Train.csv', index=False)
    Valid.to_csv('/home/guoj5/Desktop/Valid.csv', index=False)

#   train: test: validation = 7:2:1

###################### I am here : curation on the hard negative samples ##############################################
## 1. track all positive samples that already have
all_positive_filtered = pd.read_csv('/home/guoj5/Desktop/all_positive_filtered.csv')  # len(): 1634
all_negative_filtered = pd.read_csv('/home/guoj5/Desktop/all_negative_filtered.csv')  # len(): 79920

########################################################################################################################




# 2. split of the number of positive samples
num_train_pos = int(0.7 * len(all_positive_filtered))
num_test_pos = int(0.2 * len(all_positive_filtered))
num_valid_pos = int(0.1 * len(all_positive_filtered))


# down sample
train_pos = all_positive_filtered.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:num_train_pos]
valid_pos = all_positive_filtered.sample(frac=1, random_state=42).reset_index(drop=True).iloc[num_train_pos: num_train_pos + num_valid_pos]
test_pos = all_positive_filtered.sample(frac=1, random_state=42).reset_index(drop=True).iloc[num_train_pos + num_valid_pos : num_train_pos + num_valid_pos + num_test_pos]


# 3. sample from the negative samples (re-balancing / downsampling)
############################### I am here: Sample from the Negative samples ##################################################
# shuffle negative samples
all_negative_filtered = all_negative_filtered.sample(frac=1, random_state=42).reset_index(drop=True)
train_neg = all_negative_filtered.iloc[:1500]
valid_neg = all_negative_filtered.iloc[1500:1500 + 10 * len(valid_pos)]
########################################################################################################################

# concat pos and neg for dataset
Train = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
Valid = pd.concat([valid_pos, valid_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

Train.to_csv('/home/guoj5/Desktop/Train.csv', index=False)
Valid.to_csv('/home/guoj5/Desktop/Valid.csv', index=False)
