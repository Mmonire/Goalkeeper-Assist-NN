# Goalkeeper Assist Neural Network Project

## Overview
This project develops a deep neural network using **Keras** to assist the goalkeeper of the French national football team in making successful ball throws to midfield. The dataset is small, leading to potential overfitting, so the project employs **L1L2 regularization** and **Dropout** techniques to improve model generalization. The goal is to predict whether a thrown ball will reach a teammate (successful throw) or an opponent (unsuccessful throw) based on player positions on the field.

The project is implemented in a Jupyter Notebook (`goalkeeper.ipynb`) and includes data loading, preprocessing, model construction with regularization, training, evaluation, and visualization of decision boundaries. The final output includes model architecture details and predictions, packaged for submission.

## Project Structure
The repository contains the following files:
- **goalkeeper.ipynb**: The main Jupyter Notebook containing the complete workflow, including data loading, preprocessing, model building, training, and visualization.
- **model_l1l2_info.json**: JSON file describing the architecture of the neural network with L1L2 regularization.
- **model_dropout_info.json**: JSON file describing the architecture of the neural network with Dropout regularization.
- **README.md**: This file, providing an overview and instructions for the project.

## Usage
The project is executed within the `goalkeeper.ipynb` Jupyter Notebook. Below is an overview of the workflow:

### 1. Data Loading
- Training and validation datasets are loaded using `np.load` from `data_X.npy`, `data_y.npy`, `data_Xval.npy`, and `data_yval.npy`.
- The dataset contains two features (likely representing 2D coordinates on the football field) and a binary label (0 for opponent, 1 for teammate).

### 2. Data Preprocessing
- The features are already in a suitable format (2D coordinates) and require no additional preprocessing.
- The dataset is small (211 training samples, 200 validation samples), increasing the risk of overfitting, which is addressed through regularization.

### 3. Model Architecture
Two models are implemented to combat overfitting:
- **L1L2 Regularization Model (`model_l1l2`)**:
  - Dense layers with L1 and L2 regularization applied to the kernel weights to penalize large weights and encourage sparsity.
  - Architecture details are saved in `model_l1l2_info.json`.
- **Dropout Regularization Model (`model_dropout`)**:
  - Dense layers interspersed with Dropout layers to randomly deactivate a fraction of neurons during training, reducing co-dependency.
  - Architecture details are saved in `model_dropout_info.json`.
- Both models use a sigmoid activation in the output layer for binary classification.

### 4. Model Training
- Models are compiled with:
  - Optimizer: **Adam**
  - Loss function: **BinaryCrossentropy**
  - Metric: **Accuracy**
- Training is performed on the training dataset, with performance monitored on the validation set.

### 5. Visualization
- The decision boundaries of the `model_l1l2` are visualized using a custom `plt_decision_boundaries` function.
- Training data points are plotted, with blue indicating successful throws (to teammates) and red indicating unsuccessful throws (to opponents).

## Evaluation
- The models' performance is evaluated using accuracy on the validation set.
- The use of L1L2 and Dropout regularization improves generalization compared to a baseline model, as observed in the decision boundary visualization.
- The project aims to achieve better generalization by reducing overfitting, which is critical given the small dataset size.

## Results
- The L1L2 and Dropout models demonstrate improved performance over a non-regularized model, with smoother decision boundaries and higher validation accuracy.
- The visualization shows that the models effectively separate successful and unsuccessful throws, indicating good generalization.

## Notes
- The dataset is small, making regularization essential to prevent overfitting.
- The L1L2 model uses both L1 (lasso) and L2 (ridge) regularization to control model complexity.
- The Dropout model randomly deactivates neurons, enhancing robustness.
- Future improvements could include hyperparameter tuning (e.g., regularization strength, dropout rate) or experimenting with other regularization techniques like batch normalization.
