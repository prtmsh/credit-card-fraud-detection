# Credit Card Fraud Detection

## Overview

This project focuses on detecting fraudulent credit card transactions using machine learning. The dataset used for this project is an anonymized collection of credit card transactions from European cardholders made in September 2013. The aim is to develop a model that can accurately distinguish between fraudulent and genuine transactions, helping to prevent unauthorized charges on customers' accounts.

## Dataset

### Description
- **Source:** An anonymized dataset containing credit card transactions.
- **Time Period:** Transactions made over two days in September 2013.
- **Size:** 284,807 transactions with 492 labeled as frauds (0.172% of total).
- **Features:**
  - **V1 to V28:** Principal components obtained from PCA transformation.
  - **Time:** Seconds elapsed between each transaction and the first transaction in the dataset.
  - **Amount:** The transaction amount, which can be used for cost-sensitive learning.
  - **Class:** The target variable (1 indicates fraud, 0 indicates genuine).

### Data Imbalance
The dataset is highly imbalanced, with the positive class (fraudulent transactions) accounting for only 0.172% of the total transactions. Due to this imbalance, performance evaluation using standard accuracy metrics is not meaningful. Instead, metrics like the Area Under the Precision-Recall Curve (AUPRC) are recommended.

## Project Structure

### 1. Data Preprocessing (`notebooks/01_data_preprocessing.ipynb`)
- **Data Loading:** The dataset is loaded and initial exploratory data analysis is performed.
- **Missing Values:** Checked for missing values in the dataset.
- **Data Visualization:**
  - Distribution of the target class (`Class`).
  - Correlation matrix of the features.
- **Feature Scaling:** `Time` and `Amount` features are scaled using `StandardScaler`.
- **Data Export:** Preprocessed data is saved as `processed_data.csv` for model training.

### 2. Model Training (`notebooks/02_model_training.ipynb`)
- **Data Splitting:** The preprocessed dataset is split into training and testing sets (80/20 split).
- **Model Selection:** Random Forest Classifier is used as the model for fraud detection.
- **Model Training:** The model is trained on the training data and saved for future use.
- **Model Evaluation:**
  - **Classification Report:** Precision, recall, f1-score, and support for each class.
  - **Confusion Matrix:** Displaying the number of true positives, true negatives, false positives, and false negatives.
  - **ROC Curve:** Visual representation of the trade-off between true positive rate and false positive rate.
- **Model Export:** Trained model is saved as a pickle file (`random_forest_model.pkl`).
- **Evaluation Report:** Classification report, confusion matrix, and AUC score are saved to `evaluation_report.txt`.

## How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/prtmsh/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Install Dependencies:**
   Make sure you have all necessary libraries installed. You can use the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preprocessing:**
   Run the data preprocessing notebook to prepare the dataset.
   ```bash
   jupyter notebook notebooks/01_data_preprocessing.ipynb
   ```

4. **Model Training:**
   Run the model training notebook to train the Random Forest model.
   ```bash
   jupyter notebook notebooks/02_model_training.ipynb
   ```

5. **Model Evaluation:**
   After training, you can check the saved evaluation report (`outputs/evaluation_report.txt`) and ROC curve (`outputs/roc_curve.png`) for model performance.

## Results

- **Classification Report:** Displays precision, recall, and f1-score for both fraud and genuine transactions.
- **Confusion Matrix:** Provides an overview of the model's prediction performance.
- **AUC Score:** Evaluates the model's ability to distinguish between classes.
- **ROC Curve:** Shows the performance of the classifier across different threshold values.

## References

- [Kaggle Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Practical Handbook on Machine Learning for Credit Card Fraud Detection](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Chapter_3_GettingStarted/SimulatedDataset.html)