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

## Prerequisites
To run the project, you need the following dependencies:
- **Python 3.x**
- **Required Libraries**:
  ```bash
  pip install numpy matplotlib tensorflow
  ```
- **Input Files**:
  - `data_X.npy`: Training features.
  - `data_y.npy`: Training labels.
  - `data_Xval.npy`: Validation features.
  - `data_yval.npy`: Validation labels.
  Ensure these files are placed in a `data` directory relative to the notebook.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required Python packages:
   ```bash
   pip install numpy matplotlib tensorflow
   ```
3. Ensure the input `.npy` files are in the `data` directory.

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

### 6. Submission
- The notebook generates `model_l1l2_info.json` and `model_dropout_info.json`, which describe the architectures of the respective models.
- A `result.zip` file is created containing:
  - `model_l1l2_info.json`
  - `model_dropout_info.json`
  - `goalkeeper.ipynb`
- The submission cell uses the following code to package the files:
  ```python
  import zipfile
  import joblib

  if not os.path.exists(os.path.join(os.getcwd(), 'goalkeeper.ipynb')):
      %notebook -e goalkeeper.ipynb

  def compress(file_names):
      print("File Paths:")
      print(file_names)
      compression = zipfile.ZIP_DEFLATED
      with zipfile.ZipFile("result.zip", mode="w") as zf:
          for file_name in file_names:
              zf.write('./' + file_name, file_name, compress_type=compression)

  file_names = ['model_l1l2_info.json', 'model_dropout_info.json', 'goalkeeper.ipynb']
  compress(file_names)
  ```

### Running the Notebook
To execute the project:
1. Open `goalkeeper.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure the required libraries are installed and the input `.npy` files are in the `data` directory.
3. Run all cells in the notebook to perform data loading, model training, visualization, and file generation.
4. The final output (`result.zip`) will be generated in the working directory.

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

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.