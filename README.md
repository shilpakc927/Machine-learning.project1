# 📞📊 Telecom Customer Churn Prediction using Logistic Regression

This project demonstrates a **machine learning classification task** using **Logistic Regression** to predict whether telecom customers are likely to churn (leave the service). The dataset used is `telecom_churn.csv`.

---

## 📂 Project Structure

```
├── telecom_churn.csv          # Dataset containing customer features and churn labels
├── ml_project1.ipynb          # Jupyter Notebook with complete code and analysis
├── README.md                  # Project overview and instructions
```

---

## 📝 Dataset Description

**File:** `telecom_churn.csv`

This dataset contains **tabular data** with customer attributes and a churn label indicating whether the customer has left the company.  
Each row represents a single customer record, including:

- **Demographic Data:** e.g., age, gender, location
- **Service Usage:** e.g., call minutes, data usage
- **Account Information:** e.g., tenure, contract type, payment method
- **Label Column:** `Churn` (1 if the customer churned, 0 if retained)

Example (illustrative):

| CustomerID | Tenure | MonthlyCharges | Contract   | ... | Churn |
|------------|--------|----------------|------------|-----|-------|
| C001       | 12     | 70.50          | Month-to-Month | ... | 0     |
| C002       | 5      | 85.20          | Two Year       | ... | 1     |

---

## 🧠 Algorithm Overview

**Logistic Regression** is a supervised classification algorithm that models the probability of a binary outcome:

- Calculates the **log-odds** of the positive class.
- Uses the **sigmoid function** to transform predictions into probabilities.
- Thresholds probabilities (e.g., 0.5) to assign class labels.

Advantages:
- Interpretable model coefficients
- Simple and efficient
- Works well for linearly separable data

Considerations:
- May underperform on complex non-linear relationships without feature engineering

---

## 🚀 Project Workflow

The project follows these main steps:

1. **Import Libraries**
   - `pandas`, `numpy`: Data manipulation
   - `scikit-learn`: Model building and evaluation
   - `matplotlib`, `seaborn`: Visualization

2. **Load and Explore Data**
   - Read `telecom_churn.csv`
   - Inspect data shape, types, and churn distribution

3. **Preprocess Data**
   - Encode categorical variables (e.g., One-Hot Encoding)
   - Handle missing values if present
   - Scale numerical features

4. **Split Data**
   - Training set and test set (e.g., 80/20 split)

5. **Train Logistic Regression Model**
   - Fit model to training data
   - Extract model coefficients

6. **Evaluate Model**
   - Predict churn probabilities and labels
   - Compute metrics:
     - Accuracy
     - Confusion matrix
     - Precision, recall, F1-score
     - ROC-AUC

7. **Visualize Results**
   - Plot confusion matrix
   - Plot ROC curve

---

## ⚙️ How to Run

Follow these steps to run the project:

1. **Install Requirements**

   Make sure you have Python >=3.7 and install dependencies:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

2. **Open Notebook**

   Launch Jupyter Notebook:

   ```bash
   jupyter notebook ml_project1.ipynb
   ```

3. **Run Cells**

   Execute each cell sequentially to reproduce preprocessing, training, and evaluation.

---

## 🎯 Results

Your model performance will depend on:

- Data preprocessing
- Feature encoding
- Threshold selection

Example metrics (hypothetical):

- **Accuracy:** 82%
- **Precision:** 78%
- **Recall:** 65%
- **F1-score:** 71%
- **ROC-AUC:** 0.85

---

## 🔧 Tuning & Improvements

To improve your classifier:

- Experiment with regularization (`L1`, `L2`)
- Adjust classification threshold
- Balance classes (e.g., SMOTE)
- Engineer new features (e.g., tenure buckets)
- Try alternative models (Random Forest, XGBoost)

---

## 📈 Sample Code Snippet

Example code to train Logistic Regression:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Load dataset
data = pd.read_csv("telecom_churn.csv")

# Preprocess: encode, scale, etc.
# Example:
X = data.drop("Churn", axis=1)
y = data["Churn"]

# One-hot encoding of categorical columns (if needed)
X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Evaluate
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, logreg.predict_proba(X_test)[:, 1])

print("Accuracy:", accuracy)
print("ROC-AUC:", roc_auc)
```

---

## 📚 References

- [Scikit-learn Documentation – LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Logistic Regression Explained](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)

---

## 🙌 Acknowledgements

This project was created as part of a machine learning assignment to practice churn prediction using Logistic Regression.

---

## 📩 Contact

For questions or suggestions, feel free to reach out.

