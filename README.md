# Spark ML Pipeline: Hyperparameter Tuning with GBT

This repository demonstrates how to build a machine learning pipeline using **Apache Spark** and **PySpark** for classification tasks. The focus is on applying **Gradient Boosted Trees (GBT)** for predicting subscriber status, with **hyperparameter tuning** and **cross-validation** to optimize model performance.

## Project Overview

The pipeline includes several key steps:
- **Feature Engineering**: Includes transformations for categorical variables (using **StringIndexer** and **OneHotEncoder**) and scaling for continuous features (using **StandardScaler**).
- **Model Setup**: A **Gradient Boosted Trees (GBT)** classifier is used for classification.
- **Hyperparameter Tuning**: A **cross-validation** strategy is used to fine-tune hyperparameters such as `maxIter` and `stepSize`.
- **Model Evaluation**: The model is evaluated using **BinaryClassificationEvaluator** to assess its performance on the test data.

## Key Components
- **Data Preprocessing**: Transforming features, scaling continuous data, and one-hot encoding categorical variables.
- **GBT Classifier**: A powerful tree-based model used for binary classification tasks.
- **Cross-Validation**: Optimizing hyperparameters using cross-validation to improve generalization and avoid overfitting.

## Results
- The trained model predicts whether a user is a subscriber based on various features.
- Evaluation metrics such as **raw predictions**, **probabilities**, and final **predictions** are displayed for the test data.

## Libraries Used
- **PySpark** for distributed data processing and machine learning.
- **MLlib** for scalable machine learning algorithms.
