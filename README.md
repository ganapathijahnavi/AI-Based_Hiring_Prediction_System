# AI-Based_Hiring_Prediction_System

## Project Overview

This project builds an end-to-end machine learning system that predicts whether a candidate will be hired or rejected based on resume-related features. The system simulates an AI-powered resume screening tool used in HR analytics.

The model evaluates candidate information such as skills, experience, education, certifications, projects, and salary expectations to generate a hiring decision along with a probability score.

---

## Problem Statement

Manual resume screening is time-consuming and subjective. The goal of this project is to design a machine learning pipeline that automates candidate evaluation and predicts hiring decisions based on structured resume data.

Target Variable:
- Hire → 1  
- Reject → 0  

---

## Dataset Description

The dataset contains 1000+ synthetic resumes with the following features:

- Skills (text)
- Experience (Years)
- Education
- Certifications (text)
- Job Role (text)
- Salary Expectation
- Projects Count
- Recruiter Decision (Target)

Important:
The AI Score column was intentionally removed to prevent data leakage and ensure independent model learning.

---

## Project Workflow

### 1. Data Cleaning
- Removed identifier columns
- Handled missing values in certifications
- Converted target variable into numeric format

### 2. Text Feature Engineering
- Combined skills and certifications into a single text feature
- Applied text cleaning (lowercase, remove special characters)
- Used TF-IDF vectorization

### 3. Feature Processing
- Scaled numerical features using StandardScaler
- Encoded education using Label Encoding
- Applied stratified train-test split due to class imbalance

### 4. Model Training
Four models were trained and compared:
- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors

### 5. Hyperparameter Tuning
Used GridSearchCV with 5-fold cross-validation to tune Logistic Regression regularization parameter.

---

## Best Model Performance

- Model: Logistic Regression
- Best C value: 10
- Cross-validation F1 Score: 0.988
- Test Accuracy: 98.5%

The dataset was highly structured and linearly separable, which explains the strong performance of Logistic Regression compared to more complex models.

---

## Production-Ready Pipeline

A complete scikit-learn Pipeline was implemented to ensure:
- Consistent preprocessing
- No data leakage
- Reproducibility
- Deployment readiness

The trained pipeline was saved using joblib for future use.

---

## Hiring Prediction Function

A prediction function was created that:
- Accepts candidate details
- Applies preprocessing automatically
- Returns hiring decision
- Returns probability score

Example Output:

Decision: Hire  
Probability: 0.997  

---

## Project Structure
AI-Hiring-Prediction-System/
│
├── data/
├── notebook
├── requirements.txt
└── README.md

---

## Key Learnings

- Importance of preventing data leakage
- Handling class imbalance using stratified splitting
- Text feature engineering using TF-IDF
- Model comparison and evaluation beyond accuracy
- Hyperparameter tuning using GridSearchCV
- Building deployment-ready ML pipelines

---

## Real-World Application

This system demonstrates how AI can support HR automation by:
- Reducing manual resume screening effort
- Providing consistent and data-driven hiring support
- Assisting recruiters with probability-based decision insights

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorization
- GridSearchCV

---

## Future Improvements

- Handle unseen categories more robustly
- Apply imbalance handling techniques such as SMOTE
- Deploy as a Streamlit web application
- Add explainability using SHAP or LIME

---

## Author

Jahnavi Durga Ganapathi 
AI / Machine Learning Enthusiast

