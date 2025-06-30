
# Telecom Churn Prediction - Logistic Classification

This project uses **Logistic Regression** to predict customer churn based on a telecom dataset. The model identifies patterns in customer behavior to determine the likelihood of churn, enabling proactive retention strategies.

---

## 📁 Dataset
**Name**: `telecom_churn.csv`  
The dataset contains information on customer usage, account details, service plans, and whether they have churned.

### Key Features:
- State, Area Code
- Account Length
- International Plan
- Voice Mail Plan
- Number of Voice Mail Messages
- Total Day/Eve/Night/Intl Minutes & Charges
- Customer Service Calls
- Target: **Churn**

---

## ⚙️ Tools & Libraries
- Python
- pandas, numpy
- scikit-learn
- StandardScaler (for normalization)
- LogisticRegression
- Accuracy Score, Confusion Matrix, Classification Report

---

## 🔍 Model Workflow

```python
# Load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score
from sklearn.preprocessing import StandardScaler
```

1. **Data Preprocessing**
   - Handling categorical data (e.g., `International Plan`, `Voice Mail Plan`)
   - Scaling numerical features

2. **Model Training**
   - Split into train/test sets
   - Train logistic regression on the scaled features

3. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

---

## ✅ Results

The model performs binary classification to predict churn. Evaluation metrics help in assessing precision, recall, and F1-score, indicating the model's effectiveness in detecting churned customers.

---

## 📌 Conclusion

This project demonstrates how logistic regression can be applied to real-world customer datasets to provide actionable insights. With proper feature engineering and scaling, the model yields reliable predictions.

---

