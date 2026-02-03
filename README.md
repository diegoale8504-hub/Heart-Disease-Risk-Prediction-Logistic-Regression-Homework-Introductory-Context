# Heart-Disease-Risk-Prediction-Logistic-Regression-Homework-Introductory-Context
Repositorio para taller 2
# Heart Disease Risk Prediction: Logistic Regression Homework

This project implements **Logistic Regression from scratch (using NumPy only)** to predict heart disease risk based on clinical features. The workflow includes Exploratory Data Analysis (EDA), model training, decision boundary visualization, L2 regularization, and preparation for deployment in Amazon SageMaker.  

The objective is to simulate a production-ready machine learning pipeline while reinforcing theoretical concepts such as sigmoid function, binary cross-entropy loss, gradient descent optimization, and regularization.

---

# Getting Started

These instructions will help you run the project locally for development and testing purposes. Deployment notes for Amazon SageMaker are included in the Deployment section.

---

# Prerequisites

You need the following installed:

- Python 3.9 or higher
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib

You can install everything using:

Run All Cells

Run the notebook from top to bottom.

The notebook will:

Perform Exploratory Data Analysis (EDA)

Encode the target variable (Presence → 1, Absence → 0)

Normalize numerical features

Perform 70/30 stratified train-test split

Implement Logistic Regression from scratch

Train using Gradient Descent

Plot cost convergence

Evaluate performance metrics

Visualize decision boundaries

Apply L2 regularization

Compare regularized vs unregularized models

unning the Tests

This project evaluates correctness through end-to-end model validation.

End-to-End Evaluation

The complete ML pipeline is validated by:

Splitting data (70/30 stratified)

Training logistic regression

Generating predictions

Computing evaluation metrics

Example:

train_metrics = evaluate(y_train, y_train_pred)
test_metrics = evaluate(y_test, y_test_pred)


These tests verify:

- Correct gradient computation

- Proper cost convergence

- Generalization performance on unseen data

Coding Style Checks

## Code follows:

Clear mathematical implementation aligned with theory

Modular function structure (sigmoid, cost, gradient, predict)

Inline comments explaining each step

Separate sections for EDA, training, visualization, and regularization

## Deployment

The trained model can be deployed using Amazon SageMaker.

Deployment Steps Overview

Upload notebook to SageMaker Studio.

Train the model in SageMaker.

Save learned parameters (w, b) as .npy files.

Create inference script that:

Loads saved weights

Accepts patient feature input

Returns probability of heart disease.

Deploy endpoint.

Test using sample patient data.

Example Test Input
Age = 60
Cholesterol = 300
BP = 140
Max HR = 150
ST depression = 2.3
Number of vessels = 1

Predicted Probability: 0.68
Risk Level: High

Built With:
- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- Amazon SageMaker

## Versioning

Version 1.0 — Initial implementation including:
- EDA
- Logistic regression from scratch
- Decision boundary visualization
- L2 regularization
- Deployment documentation

Authors
Diego Rozo — Implementation and analysis

License
This project is intended for academic use only.

## Acknowledgments

- World Health Organization (cardiovascular statistics context)
- Kaggle (Heart Disease dataset)
- Course materials on Logistic Regression and Regularization
- AWS SageMaker documentation
