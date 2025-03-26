# ImageClassificationCNN

# Injury Classification Using CNN

## Overview
This project focuses on classifying injuries using a Convolutional Neural Network (CNN). The model is trained on images categorized into three classes: **hand injuries, head injuries, and leg injuries**. The pipeline includes data preprocessing, model training, evaluation, and testing on unseen images.

## Project Structure
- **Data Preparation**: Loading and cleaning image datasets.
- **Data Preprocessing**: Scaling and splitting the data into training, validation, and test sets.
- **Model Development**: Implementing a CNN using TensorFlow and Keras.
- **Training and Evaluation**: Training the model and assessing its performance.
- **Predictions on Unseen Data**: Testing the model with new images.
- **Model Saving**: Storing the trained model for future use.

## Installation
To set up the project, install the required dependencies:
```sh
pip install tensorflow opencv-python numpy
```
Ensure TensorFlow and OpenCV are installed correctly before proceeding.

## Data Preparation
- Load images from the dataset directory.
- Remove unwanted system files (e.g., `.DS_Store`).
- Filter out non-image files based on their extensions.

## Model Architecture
The CNN consists of:
- **Convolutional Layers**: Feature extraction using filters.
- **MaxPooling Layers**: Reducing image dimensions.
- **Flattening Layer**: Transforming data for dense layers.
- **Dense Layers**: Fully connected layers for classification.
- **Softmax Activation**: Assigning probabilities to classes.

## Training the Model
The model is compiled with:
- **Optimizer**: Adam (adaptive learning rate adjustments)
- **Loss Function**: Sparse Categorical Crossentropy (measuring classification accuracy)
- **Metrics**: Accuracy, Precision, and Recall

Training runs for **15 epochs**, with validation data monitoring and learning rate reduction upon stagnation.

## Evaluation and Testing
- Model performance is assessed using precision, recall, and accuracy.
- Predictions are made on unseen images after resizing and scaling.

## Saving the Model
The trained model is saved for future inference:
```sh
model.save('models1/injury_classification')
```

## Usage
To use the trained model for classification:
1. Load the saved model.
2. Preprocess an input image (resize and scale).
3. Predict the injury type using the model.

## Future Improvements
- Expand the dataset for better generalization.
- Optimize model architecture for higher accuracy.
- Deploy the model as a web app for real-time classification.

---
### Author
Lisella Are

### License
This project is for educational purposes. Modify and use as needed.

