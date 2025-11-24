# This project applies computer vision techniques to classify chest X-ray images as NORMAL or PNEUMONIA using the Kaggle Chest X-Ray dataset. The goal is to demonstrate end-to-end understanding of preprocessing, model training, evaluation, and prediction.

1. Problem Overview

Pneumonia is a major health concern, and X-ray analysis requires expert interpretation. This project uses deep learning to support faster, consistent, and automated detection.

2. Objectives

Build a complete CV pipeline from scratch
Preprocess medical images for model training
Train and evaluate a CNN on the X-ray dataset
Generate predictions on unseen images
Present results with visual outputs

3. Dataset

Source: Kaggle – Chest X-Ray Pneumonia Dataset
Structure includes:

/train
    /NORMAL
    /PNEUMONIA
/val
/test

4. Project Structure
project/
│── src/
│     ├── cv_preprocessing.py
│     ├── train_model.py
│     ├── predict.py
│     └── show_cv_results.py
│── dataset/
│── output/           # processed images + model outputs
│── screenshot/       # UI / terminal screenshots
│── README.md
│── .gitignore

5. Tools & Technologies

Python
TensorFlow / Keras
OpenCV
NumPy, Matplotlib
VS Code
Git & GitHub

6. Methodology
# Preprocessing
CLAHE enhancement
Noise removal
Thresholding
Edge detection

# Model Training
Validation + testing

# Prediction
Load trained model
Preprocess user-selected image


7. Results

Visual results (enhancements, edges, segmented images) are saved inside:

/output


Screenshots of execution are inside:

/screenshot