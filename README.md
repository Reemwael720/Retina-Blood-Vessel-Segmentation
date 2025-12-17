# Retina-Blood-Vessel-Segmentation

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/) 
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## DESCRIPTION

This project focuses on **automatically detecting and segmenting blood vessels in retinal fundus images**, which is crucial for early diagnosis and monitoring of ophthalmic diseases such as **diabetic retinopathy, glaucoma, and hypertension-related retinal damage**.  

The approach uses a **U-Net convolutional neural network**, which is widely used in medical image segmentation due to its **encoder-decoder architecture with skip connections**, allowing the model to capture both **global context** and **fine details**.  

Key steps include:

- **Data analysis:** Compute image and mask statistics, sharpness, brightness, and channel means.  
- **Preprocessing:** Enhance green channel, apply CLAHE and unsharp masking, normalize images.  
- **Data augmentation:** Random rotation, flip, zoom, and shift to increase dataset variability.  
- **Model training:** Train U-Net with Focal Tversky Loss to handle imbalanced classes.  
- **Evaluation:** Assess performance using Dice, IoU, accuracy, precision, recall, F1-score, AUC, and visualize segmentation results.  

This project can be extended to assist in **clinical decision support systems** for ophthalmology.

---
## Table of Contents

1. [Dataset](#dataset)  
2. [Data Preprocessing](#data-preprocessing)  
3. [Data Augmentation](#data-augmentation)  
4. [Model Architecture](#model-architecture)
5. [Training](#training)
7. [Loss Function & Metrics](#loss-function--metrics)  
8. [Evaluation](#evaluation)  
9. [Visualization](#visualization)
10. [Usage](#usage)  

---
## Dataset

The dataset is divided into **train** and **test** sets:
train/
    image/
    mask/
test/
    image/
    mask/
- **Images:** Retinal fundus images (RGB).  
- **Masks:** Binary masks representing blood vessels.  

Statistics such as image dimensions, mask dimensions, brightness, sharpness, noise, and channel means are computed for analysis.

---
## Data Preprocessing

Each image undergoes the following steps:

1. Convert BGR → RGB  
2. Extract the **green channel**  
3. Apply **CLAHE** for contrast enhancement  
4. Apply **Unsharp Masking** for edge enhancement  
5. Normalize pixel values to `[0,1]`  
6. Expand dimensions for model input
7. Masks are binarized and reshaped to match the model input.

[<p align="center">
  <img src="[![Uploading preprocessing.png…]()](https://github.com/Reemwael720/Retina-Blood-Vessel-Segmentation/blob/main/code/preprocessing.png)" width="350"/>
</p>](https://github.com/Reemwael720/Retina-Blood-Vessel-Segmentation/blob/main/code/preprocessing.png)

---

## Data Augmentation

Augmentation techniques include:

- Random rotation (±15°)  
- Random flipping (horizontal & vertical)  
- Random zoom (±10%)  
- Random shift (±10% of width/height)  

Augmented images improve model generalization.


---

## Model Architecture

The **U-Net** model consists of:

- **Encoder:** 4 blocks of `Conv2D` + `MaxPooling2D`  
- **Bottleneck:** 2 `Conv2D` layers  
- **Decoder:** 4 blocks of `UpSampling2D` + skip connections  
- **Output:** 1 channel with `sigmoid` activation  


---

## Training

- Optimizer: **Adam (1e-4)**  
- Epochs: 100  
- Steps per epoch: `len(train_images)/batch_size`  
- Callbacks: `ModelCheckpoint` and `ReduceLROnPlateau`  

---
## Loss Function & Metrics

**Loss:** Focal Tversky Loss  
focal_tversky_loss_alpha_beta(alpha=0.3, beta=0.7, gamma=0.75)

**Metrics:**  
- Dice Coefficient  
- Intersection over Union (IoU)  
- Accuracy

  ---

## Evaluation

- Classification metrics: Precision, Recall, F1-score, AUC  
- Confusion matrix visualized with Seaborn  
- ROC curve plotted  


---

## Visualization

Random predictions are visualized with four views:

1. Original image  
2. Ground truth mask  
3. Predicted mask  
4. Morphological erosion of predicted mask  
 

---

## Usage

1. Set dataset paths and load images.  
2. Preprocess images and masks.  
3. Create training and validation generators.  
4. Train the U-Net model with `model.fit()`.  
5. Evaluate metrics and visualize predictions.

---
