from Custom_dataset import CustomImageDataset
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
import matplotlib.pyplot as plt



def shuffle_image_patches(image, patch_size):
    # Divide the image into patches
    h, w = image.shape[:2]
    patches = []
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append(patch)

    # Randomly shuffle the patches
    np.random.shuffle(patches)

    # Stitch the patches back together
    shuffled_image = np.zeros_like(image)
    idx = 0
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            shuffled_image[y:y+patch_size, x:x+patch_size] = patches[idx]
            idx += 1

    return shuffled_image


import numpy as np
import cv2


def detect_edges(multi_band_image):
    # Initialize an empty array to store the maximum edge values
    edge_detected = np.zeros(multi_band_image.shape[1:], dtype=np.float32)

    # Iterate through each band
    for band in multi_band_image:
        # Apply Sobel edge detection in x and y direction
        sobelx = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=5)

        # Combine the horizontal and vertical edges
        edge = np.hypot(sobelx, sobely)

        # Update the maximum edge values
        edge_detected = np.maximum(edge_detected, edge)

    # Normalize the result to the range [0, 255] for viewing
    edge_detected = cv2.normalize(edge_detected, None, 0, 255, cv2.NORM_MINMAX)

    return edge_detected.astype(np.uint8)


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
        sobelx = cv2.Sobel(band, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(band, cv2.CV_64F, 0, 1, ksize=3)

        # Combine the horizontal and vertical edges
        edge = np.hypot(sobelx, sobely)

        # Update the maximum edge values
        edge_detected = np.maximum(edge_detected, edge)

    # Normalize the result to the range [0, 255] for viewing
    edge_detected = cv2.normalize(edge_detected, None, 0, 255, cv2.NORM_MINMAX)

    return edge_detected.astype(np.uint8)



data_augmentation_transform = A.Compose([
        A.Flip(),
        A.RandomRotate90(),
        # A.Sharpen(p=1),
        # A.Solarize(threshold=0.1),
        # A.GaussianBlur(p=1)
        # A.ShiftScaleRotate(p=0.5)         # do
        # A.Transpose(p=1)                  # do , transpose row and col
        A.RandomGamma(p = 0),
        # A.RandomResizedCrop(256, 256, p = 0.5)

    ])
data_augmentation_transform = None


dataset = CustomImageDataset(data_file='/home/guoj5/Desktop/Train.csv', transform=data_augmentation_transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

count = 0
for i, (img, label) in enumerate(dataset):
    if label == 0:
        continue

    print(label)
    print(img.dtype)
    print(img.shape)

    img = img.permute(1, 2, 0)


    TEST_IMG = img.numpy()[:,:,(4, 2,  1)]  # rgb format
    edge_image = detect_edges_updated(TEST_IMG)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(TEST_IMG)
    ax[1].imshow(edge_image)
    plt.show()

    # for j in range(img.size(2)):
    #     image = img.numpy()[:, :, j]
    #     plt.imshow(image)
    #     plt.title(str(label.item()) + f'  band {j}')
    #     plt.show()


    count += 1
    if count % 20 == 0:
        break