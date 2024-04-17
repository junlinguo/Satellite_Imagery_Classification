## Project: Identification of ancient buildings and corrals from the WorldView multispectral satellite imagery 

This repository contains the scripts and resources used for our machine learning experiment with multispectral data. Below you can find descriptions and links to the scripts provided in this repository.

## Dataset

The dataset used in this experiment is hosted on Google Drive. You can access the data via the following link:

[Access Dataset](https://drive.google.com/drive/folders/1X5LfBMy8GR0xUs-7ZOdwQlqJMzRtxflw?usp=sharing)

## Scripts Description

### Main Scripts

- **Training and Validation Script**
  - `main.py`: Main script for training and validating the model.

- **Testing Script**
  - `test.py`: Script used for testing the model performance.

### Transfer Learning

- **Fine-tuning DINOv2**
  - `finetune_dinov2_rgb.py`: Adapt the DINOv2 model pre-trained on RGB data to our multispectral dataset.

### Data Visualization

- **Visualize Satellite Images**
  - `data_visualization.py`: Example code for visualizing satellite images from the dataset.

### Custom Dataset

- **PyTorch-like Custom Dataset Class**
  - `Custom_dataset.py`: Implementation of a custom dataset class compatible with PyTorch.

## Additional Information

More content will be added to this repository soon. Keep an eye on updates for more tools and documentation related to this project.
