import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
import random
import cv2



def detect_edges_updated(multi_band_image):
    """

    :param multi_band_image: (H, W, C) ndarray
    :return:
    """
    # Extract the number of bands
    num_bands = multi_band_image.shape[2]

    # Initialize an empty array to store the maximum edge values
    edge_detected = np.zeros(multi_band_image.shape[:2], dtype=np.float32)

    # Iterate through each band
    for band_index in range(num_bands):
        # Extract the current band
        band = multi_band_image[:, :, band_index]

        # Apply Sobel edge detection in x and y direction
        sobelx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)

        # Combine the horizontal and vertical edges
        edge = np.hypot(sobelx, sobely)

        # Update the maximum edge values
        edge_detected = np.maximum(edge_detected, edge)

    # Normalize the result to the range [0, 255] for viewing
    # edge_detected = cv2.normalize(edge_detected, None, 0, 255, cv2.NORM_MINMAX)

    # return edge_detected.astype(np.uint8)

    return edge_detected


class CustomImageDataset(Dataset):
    def __init__(self, data_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(data_file)
        self.transform = transform
        self.target_transform = target_transform

        # debug
        self.count = 0
        self.nan_imgs = []
        self.nan_labels = []

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        image = np.load(img_path)

        if self.transform:
            # the format for 'albumentations', [h w c] numpy array
            image = image.transpose(1, 2, 0)
            image = self.transform(image=image)["image"]

            # adding edges information as a new band
            new_band = detect_edges_updated(image)
            image = np.concatenate((image, new_band[:, :, np.newaxis]), axis=2)

            image = image.transpose(2, 0, 1)    # [ c h w ], torch format

        if self.target_transform:
            label = self.target_transform(label)

        if not torch.is_tensor(image):
            image = torch.tensor(image)

        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        if torch.isnan(image).any():
            # print(f'image {img_path} contains nan')
            self.count+=1
            self.nan_imgs.append(img_path)
            self.nan_labels.append(label.item())


        return image, label


if __name__ == "__main__":

    data_augmentation_transform = A.Compose(
        [
            A.Flip(),
            A.RandomRotate90()
        ]
    )
    
    # test code on a small dataset 
    dataset = CustomImageDataset(data_file='./sample_ds_5_validation.csv', transform=data_augmentation_transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    samples, labels = next(iter(dataloader))
    print(labels)
    print('done')



