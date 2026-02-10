# Heart Disease Risk Prediction: Logistic Regression from Scratch

This project implements Logistic Regression from scratch using NumPy to predict the risk of heart disease based on clinical patient data. The workflow includes Exploratory Data Analysis (EDA), feature preprocessing, gradient descent optimization, decision boundary visualization, L2 regularization, and cloud deployment using Amazon SageMaker. The objective is to simulate a real-world machine learning pipeline while reinforcing theoretical foundations such as sigmoid activation, binary cross-entropy loss, and regularized optimization.

---

# Getting Started

These instructions will help you set up and run the project locally for development and testing purposes. See the Deployment section for instructions on deploying the model to Amazon SageMaker for live inference.

---

# Prerequisites

You need the following software installed:

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab
- pip (Python package manager)

Required Python libraries:

- numpy
- pandas
- matplotlib
- scikit-learn (ONLY for train/test split, not model training)

Install dependencies with:
pip install numpy pandas matplotlib scikit-learn notebook

## Installing

Follow these steps to set up the development environment.

Step 1 — Clone the repository
git clone https://github.com/yourusername/heart-disease-lr.git
cd heart-disease-lr

## Step 2 — Ensure dataset is available

Place the dataset file:Heart_Disease_Prediction.csv

## Step 3 — Launch Jupyter Notebook
jupyter notebook
open: heart_disease_lr_analysis.ipynb

## Step 4 — Run the notebook

Run all cells from top to bottom. This will:

Perform EDA

Encode target variable (Presence → 1, Absence → 0)

Normalize selected features

Split dataset (70/30 stratified)

Train logistic regression using gradient descent

Evaluate performance metrics

Visualize decision boundaries

Apply L2 regularization

Compare regularized vs unregularized models

## Example Output

After running the notebook, you should see results similar to:
Test Accuracy: 0.76
Precision: 0.81
Recall: 0.61
F1 Score: 0.69

## Running the Tests
Disclaimer: We can not use Amazon Sage Maker because of permisions in jupyter. You will see proof that i run it in sage maker code editor:
Script:
<img width="1185" height="962" alt="image" src="https://github.com/user-attachments/assets/dd1767ae-3973-46fe-869c-e4a05dbdb234" />
<img width="1438" height="573" alt="image" src="https://github.com/user-attachments/assets/dfb54745-f382-4087-928d-27cac42327ad" />
<img width="1916" height="986" alt="image" src="https://github.com/user-attachments/assets/ba573e36-41b3-47a1-a5ab-accf72c74d83" />
<img width="1919" height="1009" alt="image" src="https://github.com/user-attachments/assets/7e2f4423-1679-4b53-b255-98d981ee366d" />
<img width="571" height="450" alt="image" src="https://github.com/user-attachments/assets/b6b0dd30-da3b-4b5c-b5f1-c88d313094e0" />
<img width="1919" height="995" alt="image" src="https://github.com/user-attachments/assets/d893e3f1-781e-4dea-8d45-33240ffa1df5" />
<img width="1919" height="1002" alt="image" src="https://github.com/user-attachments/assets/469bd18b-a1e2-4d85-9265-0c8c962c6bc1" />
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/8a9a2fe9-a63a-418c-ab3d-d573be93c38c" />
<img width="1913" height="1015" alt="image" src="https://github.com/user-attachments/assets/d22bb183-0fe3-4ec1-b0f7-1a085335bc67" />
<img width="1919" height="1002" alt="image" src="https://github.com/user-attachments/assets/1f28741d-1aed-4982-99b6-de7d7cbdd9f6" />
<img width="565" height="455" alt="image" src="https://github.com/user-attachments/assets/15addf8b-cf9a-4218-bb09-53e21ac53817" />
<img width="1919" height="1014" alt="image" src="https://github.com/user-attachments/assets/bed476eb-b444-423c-8161-8af01ac2da91" />
<img width="1919" height="1006" alt="image" src="https://github.com/user-attachments/assets/c32d3843-9fca-4727-87cc-88706de9461f" />
<img width="576" height="455" alt="image" src="https://github.com/user-attachments/assets/6067d938-af6c-484b-b045-8ec866ea53ed" />
<img width="1919" height="1014" alt="image" src="https://github.com/user-attachments/assets/cc68b04a-6313-413d-9c9c-3548a8f0e10f" />
<img width="565" height="455" alt="image" src="https://github.com/user-attachments/assets/7c4e7d81-780c-4509-97cb-c28edccf40aa" />
<img width="1919" height="1019" alt="image" src="https://github.com/user-attachments/assets/4f5aca1d-2cc6-460f-bd45-fc5f16d89741" />
<img width="1919" height="947" alt="image" src="https://github.com/user-attachments/assets/b56a3822-1ed8-45a2-9f7f-4081e8272778" />
<img width="1919" height="1024" alt="image" src="https://github.com/user-attachments/assets/d9982deb-592d-41ba-8024-2020cc63f3d1" />

## Data save
I have save this data because it was suposed to be used in the deployment in jupyter but it does not let me do it 
<img width="1919" height="959" alt="image" src="https://github.com/user-attachments/assets/a3608e85-215a-43c0-9f28-dc890d7c8749" />



Although this project does not use a formal testing framework, validation is performed through structured evaluation.

End-to-End Tests

End-to-end evaluation verifies:

-- Correct gradient computation
-- Proper convergence of cost function
-- Generalization performance on unseen data

Example evaluation:
train_metrics = evaluate(y_train, y_train_pred)
test_metrics = evaluate(y_test, y_test_pred)

These tests ensure the model:

-- Learns meaningful parameters
-- Does not diverge
-- Maintains stable performance with regularization

##   Coding Style Tests

Code quality is ensured by:

-- Modular function definitions (sigmoid, compute_cost, gradient_descent, predict)
-- Clear mathematical alignment with theory
-- Inline documentation explaining formulas
-- Separation of concerns (EDA, training, visualization, deployment)

Optional style checking:
pip install flake8
flake8 .

## Deployment

The trained model can be deployed to Amazon SageMaker as a real-time inference endpoint.

Deployment Steps Overview

Save trained weights and bias (weights.npy, bias.npy)

Create inference.py handler

Package model into model.tar.gz

Upload to Amazon S3

Deploy endpoint using SageMaker SDK

Test endpoint with patient input

Example inference request:
{
  "inputs": [[60, 140, 300, 150, 2.3, 1]]
}
Example response:
{
  "probability": [0.68]
}

## Built With

Python — Core programming language

NumPy — Numerical computation

Pandas — Data manipulation

Matplotlib — Visualization

Jupyter Notebook — Interactive experimentation

Amazon SageMaker — Cloud deployment platform

## Contributing

This project is part of an academic assignment. Contributions are currently not open, but suggestions for improvements are welcome.

##Versioning

We use Semantic Versioning (SemVer).

Current Version: 1.0.0
Initial release including EDA, logistic regression implementation, visualization, regularization, and deployment.

## Authors

Diego Rozo — Implementation, analysis, and deployment

## License

This project is for academic use only.

## Acknowledgments

World Health Organization (cardiovascular statistics context)

Kaggle — Heart Disease Dataset

UCI Machine Learning Repository

AWS SageMaker documentation

Course materials on Logistic Regression and Regularization
