# Consumer Banking Credit Risk Intelligence Platform

An end-to-end machine learning and data analytics project for analyzing consumer loan credit risk, identifying default patterns, and predicting borrower default probability.

##  Project Overview

This project analyzes consumer banking loan data to understand the factors associated with loan default and develop a machine learning model for borrower risk prediction.

The project combines:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Validation
- Feature Engineering
- SQL Analysis
- Machine Learning
- Model Evaluation
- Individual Borrower Risk Prediction
- Interactive Streamlit Dashboard

---

##  Business Problem

Financial institutions need reliable ways to identify borrowers who may have a higher probability of loan default.

This project addresses questions such as:

- Which loan grades have the highest default rates?
- How does borrower income relate to loan amount?
- Which borrower and loan characteristics are associated with higher risk?
- How does previous default history affect credit risk?
- What is the predicted default probability for an individual borrower?

---

##  Key Insights

The exploratory analysis identified several important patterns:

- Loan grade showed a strong relationship with default risk.
- Lower-income borrower groups generally had higher default rates.
- Higher loan amounts were associated with increased default risk.
- Borrowers with previous recorded defaults had substantially higher default rates.
- Higher loan-to-income burden was strongly associated with default risk.

---

##  Data Preparation

The dataset was cleaned and validated before modeling.

Key steps included:

- Duplicate removal
- Age validation
- Employment-length validation
- Missing-value treatment
- Numerical median imputation
- Categorical most-frequent imputation
- Categorical one-hot encoding

---

##  Feature Engineering

Two additional financial risk indicators were created.

### Income per Loan

```text
income_per_loan = person_income / loan_amnt
