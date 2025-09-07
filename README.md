# 🏦 Bank Transaction Categorization Using Machine Learning

This project applies supervised machine learning to automatically categorize user bank transactions based on labeled transaction history. It involves data preprocessing, feature engineering, model training, performance evaluation, and result visualization — aiming to assist users and businesses in understanding spending patterns and automating financial insights.

---

## 📁 Project Structure

```bash
├── bank_transaction_categorization_model.ipynb   # Final model notebook
├── report.ipynb                                  # Polished report notebook for presentation
├── source_code.ipynb                             # Cleaned source code version
├── model.html                                    # Exported HTML of final notebook
├── report.html                                   # Exported HTML of final report
├── README.md                                     # Project overview and usage instructions
├── bank_transaction.csv                      # Raw transaction data
├── user_profile.csv                          # User demographic/profile data

```
---

## 🧠 Problem Definition

### Task

Predict the transaction category (e.g., `Grocery`, `Entertainment`, `Utilities`, etc.) based on information such as transaction amount, date, description, and user profile.

### Inputs

- `bank_transaction.csv`: Includes transaction-level data like `description`, `amount`, `transaction date`, etc.
- `user_profile.csv`: Contains demographic data like `gender`, `income range`, `credit score`, and behavioral traits (`TRUE/FALSE` flags).

### Output

- Predicted transaction category label (class).

### Why This Matters

Accurate transaction categorization allows:
- Personalized budgeting
- Fraud detection
- Business analytics
- Auto-tagging features in fintech apps

---

## ⚙️ Model Pipeline

### 1. Preprocessing

- Cleaned `description` using regex and TF-IDF vectorizer
- Encoded categorical/boolean user profile data using one-hot encoding
- Scaled `amount` using `StandardScaler`
- Derived date-based features (weekday, weekend, month)
- Merged user and transaction data

### 2. Model Selection

- Benchmarked multiple models:
  - Logistic Regression
  - Random Forest Classifier ✅ (Chosen)
  - Gradient Boosting (planned)
- Final model: **Random Forest**, due to its robustness, interpretability, and support for heterogeneous features

### 3. Evaluation

- Used 5-fold cross-validation
- Computed macro and weighted precision/recall
- Visualized classification report and confusion matrix

### 4. Visualization

- Heatmap of classification performance across all labels
- Confusion matrix showing model error clusters
- Optional: top tokens/features by TF-IDF and importance

---

## 📊 Performance Summary

| Metric             | Score    |
|--------------------|----------|
| Macro F1-Score     | ~0.65    |
| Weighted F1-Score  | ~0.78    |
| Accuracy           | ~0.78    |

- Strong performance on categories like `Entertainment`, `Food`, `Transportation`
- Weaker performance on ambiguous/rare categories (`Misc`, class `23`)

---

## 🔍 Visual Outputs

- 📊 **Classification Heatmap**: Shows per-class precision, recall, F1-score
- 🧩 **Confusion Matrix**: Highlights overlapping or misclassified categories
- *(See `report.ipynb` or exported HTML files for rendered charts)*

---

## 🚀 Future Work

### ⏱ Next 1 Month

- Use `RandomizedSearchCV` or `GridSearchCV` for tuning
- Add `merchant_name` as a feature (if available)
- Use `SMOTE` or class weights for imbalance
- Modularize data preprocessing using custom transformers

### ⏳ Next 3 Months

- Replace TF-IDF with pretrained embeddings (e.g., FastText, BERT)
- Introduce sequence-based modeling (e.g., LSTM, Transformer)
- Build live prediction dashboard with `Streamlit`
- Integrate SHAP values for explainable ML
- Train user-specific models (e.g., by cluster or persona)

---

## 💡 Usage

1. Clone repository:
```bash
git clone https://github.com/yourusername/bank-transaction-categorization.git
cd bank-transaction-categorization
jupyter notebook bank_transaction_categorization_model.ipynb
```
2. Or view the report directly:
- model.ipynb
- report.html
