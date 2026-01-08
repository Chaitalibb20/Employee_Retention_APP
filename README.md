Employee Retention Prediction
Project Objective

This project predicts whether a data scientist is likely to look for a job change using machine learning.

Dataset

The dataset contains demographic, education, experience, and employment-related information.
The target variable indicates job change likelihood.

Machine Learning Workflow

The project follows a complete ML workflow including data exploration, preprocessing, model training, evaluation, and deployment.

Data Preprocessing

Missing values were handled using appropriate imputation techniques.
Categorical variables were encoded, and numerical features were scaled using pipelines.

Handling Class Imbalance

SMOTE was applied to address imbalance in the target variable.

Models Used

The following models were trained and compared:

Logistic Regression

Random Forest

XGBoost

LightGBM

Model Evaluation

Models were evaluated using accuracy, ROC-AUC score, and confusion matrix analysis.

Best Model

XGBoost achieved the best overall performance and was selected as the final model.

Deployment

The final model was deployed using a Streamlit web application for interactive prediction.