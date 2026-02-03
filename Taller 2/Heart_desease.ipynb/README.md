# Heart Disease Risk Prediction using Logistic Regression

## 📌 Exercise Summary
This project implements **logistic regression from scratch** to predict the presence of heart disease using clinical patient data. The objective is to demonstrate a full machine learning workflow: exploratory data analysis (EDA), model implementation without machine learning libraries, visualization of decision boundaries, regularization analysis, and a simulated cloud deployment using **Amazon SageMaker**.

The project is part of a **Machine Learning Bootcamp within a Digital Transformation and Enterprise Architecture course**, emphasizing both theoretical understanding and real-world deployment considerations.

---

## 📊 Dataset Description
- **Name:** Heart Disease Dataset  
- **Source:** Kaggle – NeuroCipher  
- **Link:** https://www.kaggle.com/datasets/neurocipher/heartdisease  
- **Records:** 303 patients  
- **Features:** 14 clinical attributes including:
  - Age (29–77 years)
  - Resting blood pressure
  - Serum cholesterol (112–564 mg/dL)
  - Maximum heart rate achieved
  - ST depression
  - Number of major vessels
- **Target Variable:**  
  - `1` → Presence of heart disease  
  - `0` → Absence of heart disease  
- **Class Distribution:** ~55% positive cases  

The dataset was downloaded manually from Kaggle and loaded as a CSV file into the project.

---

## 🧪 Methodology

### 1. Exploratory Data Analysis (EDA)
- Summary statistics and data inspection
- Class distribution visualization
- Missing value verification
- Feature selection (≥6 numerical features)
- Feature normalization
- Stratified 70/30 train-test split

---

### 2. Logistic Regression from Scratch
Implemented using **NumPy only**:
- Sigmoid function
- Binary cross-entropy loss
- Gradient descent optimization
- Parameter updates for weights and bias
- Cost tracking and convergence visualization

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score  
Computed for both training and test sets.

---

### 3. Decision Boundary Visualization
Logistic regression models were trained using **pairs of features** to visualize decision boundaries:
- Age vs. Cholesterol
- Resting BP vs. Max Heart Rate
- ST Depression vs. Number of Vessels  

Each plot includes:
- Scatter of true class labels
- Learned linear decision boundary

---

### 4. Regularization (L2)
To reduce overfitting, **L2 regularization** was added:
\[
J = J_{logistic} + \frac{\lambda}{2m} ||w||^2
\]

Tested values of λ:
