# Skin Cancer Detection with EfficientNetV2 and KerasCV

This project aims to develop and deploy a deep learning model for detecting skin cancer using 3D total body photographs. Utilizing the EfficientNetV2 backbone from KerasCV on a competition dataset, this approach integrates image data with tabular features such as age and sex to enhance prediction accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Environment Setup](#environment-setup)
- [Dataset Description](#dataset-description)
- [Data Processing](#data-processing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Inference](#inference)


## Project Overview
This project involves training a multi-input deep learning model to accurately detect skin cancer by analyzing complex data comprising both images and patient features. The backend-agnostic design allows for the use of TensorFlow, PyTorch, or JAX, leveraging KerasCV's capabilities.

## Key Features
- Uses EfficientNetV2 for high-performance image feature extraction.
- Incorporates tabular patient data to improve detection accuracy.
- Implements a robust data pipeline with random augmentations for data variability.
- Handles class imbalance through upsampling, downsampling, and class weights.
- Offers modular design for flexibility in feature selection and augmentation.
- Supports model deployment with model checkpoints for best performance.

## Environment Setup
Ensure you have the following libraries installed:
```bash
pip install tensorflow tensorflow-addons keras-cv scikit-learn pandas h5py opencv-python matplotlib
```
The project is designed to be compatible with different machine learning backends, thanks to KerasCV, which offers flexibility in choosing the preferred backend.

## Dataset Description
### Paths
- `train-image/`: Contains training image files.
- `train-image.hdf5`: Stores training image data, indexed by `isic_id`.
- `train-metadata.csv`: Metadata for training set including demographic and anatomical factors.
- `test-image.hdf5`: Stores testing image data, initially contains 3 examples, but expands to ~500,000 images upon submission.
- `test-metadata.csv`: Metadata for the testing set.

### Metadata Fields:
- `isic_id`: Unique image ID.
- `patient_id`: Unique patient ID.
- `sex`: Gender of the patient.
- `age_approx`: Approximate age.
- `anatom_site_general`: Lesion location.

## Data Processing
1. **Class Imbalance Management**: Adjust positive/negative sample proportions and use class weights.
2. **Image Loading**: Images are fetched as byte strings for memory efficiency and processed accordingly.
3. **Feature Encoding**: Tabular features are one-hot encoded and normalized as needed.
4. **Data Augmentation**: Augmentations such as RandomFlip and RandomCutout enhance training variety.

## Model Architecture
The model is comprised of two branches:
- **Image Branch**: Utilizes EfficientNetV2 B2 for feature extraction.
- **Tabular Branch**: Dense layers process additional feature inputs.
- **Concatenation and Output**: Merging branches with a final dense layer with a sigmoid for binary classification.

## Training and Evaluation
- **Loss and Metric**: Binary Crossentropy loss with class weights and ROC AUC metric are employed to guide training.
- **Learning Rate and Checkpoints**: Implements a learning rate schedule and model checkpoints ensuring that only the best models are preserved based on validation AUC.
- **Training Execution**: Model training is conducted over defined epochs, observing class weights to address imbalance, and monitors the best validation AUC.

## Inference
Upon training completion, The model achives a training accuracy of about 92%

