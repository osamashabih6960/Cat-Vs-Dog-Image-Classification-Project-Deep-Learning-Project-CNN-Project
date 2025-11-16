# Dogs vs Cats Classification Project

A deep learning project for classifying dog and cat images using Convolutional Neural Networks (CNN) with TensorFlow and Keras.

## Overview

This project implements a binary image classifier to distinguish between dogs and cats, with additional testing on breed classification. The system utilizes a CNN architecture for robust feature extraction and classification.

## Project Structure

```
.
├── dogs_cats.py        # Main implementation class
└── module10.ipynb      # Training and visualization notebook
```

## Features

**Dataset Management**
- Automated dataset organization into train/validation/test splits
- Built-in data augmentation pipeline
- Integrated TensorFlow dataset creation

**Model Architecture**
- Custom CNN implementation
- Data augmentation layers
- Optimized for binary classification

## Installation

```bash
pip install tensorflow matplotlib numpy
```

## Usage

**Dataset Preparation**
```python
from dogs_cats import DogsCats

dogs_cats = DogsCats()
dogs_cats.make_dataset_folders('validation', 0, 2400)
dogs_cats.make_dataset_folders('train', 2400, 12000)
dogs_cats.make_dataset_folders('test', 12000, 12500)
dogs_cats.make_dataset()
```

**Model Training**
```python
dogs_cats.build_network()
dogs_cats.train('model.dogs-cats')
```

**Prediction**
```python
dogs_cats.predict('path/to/image.jpg')
```

## Model Architecture

The CNN architecture includes:
- Data augmentation layers
- Convolutional layers with max pooling
- Dense layers for classification
- Binary cross-entropy loss function

## Training Process

**Callbacks Implementation**
- Early stopping for preventing overfitting
- Model checkpointing for saving best weights
- TensorBoard integration for monitoring

## Model Persistence

**Save Model**
```python
dogs_cats.save_model('model.dogs-cats')
```

**Load Model**
```python
dogs_cats.load_model('model.dogs-cats')
```

## Performance Visualization

The project includes tools for visualizing:
- Training/validation accuracy
- Loss curves
- Prediction results
