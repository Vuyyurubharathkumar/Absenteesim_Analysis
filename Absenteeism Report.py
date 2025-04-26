#!/usr/bin/env python
# coding: utf-8

# **Title:** Predictive Analysis of Employee Absenteeism Using Logistic Regression
# 
# 
# 
# ---
# 
# ## Abstract
# This report presents a comprehensive analysis of employee absenteeism data, aiming to build a predictive model for identifying excessive absenteeism using logistic regression. The study leverages data-driven techniques to enhance organizational productivity by recognizing patterns and key factors contributing to employee absence.
# 
# ---
# 
# ##  Introduction
# Absenteeism in the workplace leads to decreased productivity, higher operational costs, and disrupted workflow. Understanding and predicting absenteeism is crucial for effective workforce management. This project uses historical absenteeism data to predict excessive absenteeism and help management make informed decisions.
# 
# ---
# 
# ## Data Overview
# ###  Source and Description
# The dataset contains 700 records and 12 variables representing employee demographic details, work-related factors, and absenteeism time. Key features include:
# - **Reason for Absence** (categorical)
# - **Date** (converted to derived temporal variables)
# - **Transportation Expense, Distance to Work** (continuous)
# - **Age, Body Mass Index (BMI)** (continuous)
# - **Education, Children, Pets** (categorical)
# - **Absenteeism Time in Hours** (target variable)
# 
# ### Data Cleaning and Preparation
# Initial exploration revealed no missing values. Redundant columns like `ID` were removed. `Date` was transformed into useful components such as day of the week.
# 
# ---
# 
# ##  Preprocessing and Feature Engineering
# Preprocessing steps were carried out in the notebook `project_absenteeism_preprocessing.ipynb`:
# - **Reason for Absence:** One-hot encoded into 4 primary categories based on domain mapping.
# - **Scaling:** All numerical variables were standardized using `StandardScaler`.
# - **Feature Selection:** Variables such as `Daily Work Load Average`, `Distance to Work`, and derived temporal features were removed to prevent multicollinearity and overfitting.
# 
# ###  Target Variable Transformation
# The target variable `Absenteeism Time in Hours` was transformed into a binary classification:
# - **1 (Excessive Absenteeism):** Above median value
# - **0 (Normal Absenteeism):** At or below median
# 
# This facilitated the use of logistic regression for binary classification.
# 
# ---
# 
# ##  Model Development
# ###  Model Choice
# Logistic Regression was chosen due to its simplicity, interpretability, and effectiveness in binary classification problems.
# 
# ###  Model Training
# Training was carried out using the processed dataset:
# - **Inputs:** All standardized variables excluding the target
# - **Output:** Binary indicator for excessive absenteeism
# - **Toolkits Used:** `scikit-learn` for model development, preprocessing, and evaluation
# 
# ---
# 
# ##  Results and Analysis
# ###  Key Coefficients and Their Implications
# - **Transportation Expense (positive coefficient):** Higher cost is correlated with higher absenteeism
# - **BMI (positive coefficient):** Suggests health-related absenteeism
# - **Age (negative coefficient):** Younger employees showed slightly more absenteeism
# 
# ###  Model Evaluation Metrics
# - **Accuracy:** 77.1%
# - **Precision (Class 0 - Not Excessive):** 85.2%
# - **Recall (Class 0):** 84.1%
# - **F1-score (Class 0):** 84.7%
# - **Precision (Class 1 - Excessive):** 77.9%
# - **Recall (Class 1):** 79.3%
# - **F1-score (Class 1):** 78.6%
# - **Confusion Matrix:**
#   - True Negatives: 69
#   - False Positives: 13
#   - False Negatives: 12
#   - True Positives: 46
# 
# These metrics confirm that the model is balanced and performs reliably in identifying both excessive and normal absenteeism cases.
# 
# ---
# 
# ##  Business Implications
# ###  Actionable Insights
# - **Health Initiatives:** Consider corporate wellness programs targeting BMI management
# - **Flexible Commuting:** High transport costs suggest a need for telecommuting policies
# - **Family Policies:** Support systems for employees with children may improve attendance
# 
# ---
# 
# 
# 
# ## Random Forest 
# 1. 🎯 What does the model do?
# 
# “We’ve trained a machine learning model called a Random Forest. It looks at past patterns in data to predict whether a person is likely to [ex. be absent from work / default on a loan / make a purchase], based on various factors.”
# 
# “Our model doesn’t just make predictions — it also tells us which factors matter the most. Here's what it's learned from the data:”
# 
# 1. 🔝 Top 3 Drivers of Prediction:
# - Month Value (36.1%): “The month of the year has the biggest influence. Certain months are associated with higher or lower absenteeism.”
# 
# - Transportation Expense (12.9%): “People who spend more on commuting may be more likely to be absent — possibly due to longer distances or stress.”
# 
# - Age (9.2%): “Age has a moderate effect — potentially linked to health-related absences or family responsibilities.”
# 
# 2. ✅ How accurate is it?
# “The model is about 78.6% accurate, meaning it gets the prediction right roughly 8 out of 10 times.”
# 
# 
# - 68 times: It correctly predicted people would not (class 0).
# 
# - 42 times: It correctly predicted people would (class 1).
# 
# - 14 times: It falsely flagged someone (predicted class 1, but they were actually class 0).
# 
# - 16 times: It missed someone (predicted class 0, but they were actually class 1).
# 
# 
# ![download.png](attachment:download.png)
# 
# ---
# 
# ## 🔍 Comparing Feature Importance in **Logistic Regression vs. Random Forest**
# 
# ### ✨ Random Forest:
# - Found **`Month Value`** to be the **most important feature** (36.1% importance).
# - This is because Random Forests capture **non-linear relationships** and **interactions**.
# - Maybe certain months (like flu season, holidays, etc.) consistently have higher absenteeism — and the trees learn those patterns well.
# 
# ---
# 
# ### ➗ Logistic Regression:
# Your output shows this for `Month Value`:
# - **Coefficient**: `0.158977` (pretty low)
# - **Odds ratio (exp(coef))**: `~1.17`  
#   > For every unit increase in `Month Value`, the odds of absenteeism increase by only ~17%.
# 
# ### ➡️ Interpretation:
# > “In logistic regression, **Month Value has a small, positive but weak effect** on the odds of absenteeism — it's not statistically strong.”
# 
# ---
# 
# ## 🤔 Why the difference?
# 
# | Factor | Logistic Regression | Random Forest |
# |--------|---------------------|---------------|
# | Handles non-linearity? | ❌ No | ✅ Yes |
# | Captures feature interactions? | ❌ No | ✅ Yes |
# | Sensitive to scale? | ✅ Yes | ❌ No |
# | Simpler to interpret? | ✅ Yes | ❌ No (but still insightful via feature importance) |
# 
# ---
# 
# Here's how we can compare the **Random Forest** and **Logistic Regression** model outputs and draw conclusions based on their performance and feature importance.
# 
# ---
# 
# ### 🔍 **Model Comparison: Random Forest vs. Logistic Regression**
# 
# #### 1. **Model Performance**:
# - **Random Forest**:
#   - The model gives an overall accuracy of around **78.6%**.
#   - It captures complex relationships, including **non-linearities** and **feature interactions**.
#   - **Feature importance** shows **Month Value** as the most influential factor for absenteeism.
# 
# - **Logistic Regression**:
#   - Logistic regression's accuracy might be slightly lower due to its **linear nature** (assuming it's around 70-75% from your output).
#   - It's less flexible and cannot capture non-linear relationships or interactions between features.
#   - **Feature interpretation** is more straightforward, with **Month Value** having a weak but positive impact on absenteeism (coefficient = 0.158977).
# 
# #### 2. **Feature Importance**:
# ##### Random Forest:
# | Feature                 | Importance (%) |
# |-------------------------|----------------|
# | Month Value             | 36.1%          |
# | Transportation Expense  | 12.9%          |
# | Age                     | 9.2%           |
# | Reason_1, Reason_2, Reason_3, Reason_4 | Varying importance |
# 
# > **Random Forest Conclusion**: The model identifies **seasonality (Month Value)** as a major predictor, followed by factors like **transportation costs** and **age**. It also captures **interactions** and **non-linear patterns** that Logistic Regression can't.
# 
# ##### Logistic Regression:
# | Feature                 | Coefficient  | Odds Ratio (Exp(Coef)) |
# |-------------------------|--------------|------------------------|
# | Month Value             | 0.158977     | 1.17                   |
# | Transportation Expense  | 0.605137     | 1.83                   |
# | Age                     | -0.169906    | 0.84                   |
# | Reason_3, Reason_1, Reason_2, Reason_4 | Varying |22.51, 16.44, 2.59, 2.31
# 
# > **Logistic Regression Conclusion**: The effect of each feature is **linear**. **Month Value** has a **small positive** impact on absenteeism (odds ratio of 1.17), but the effect is not as strong as in Random Forest. **Reason_3** has a larger impact on absenteeism, indicating higher transportation costs could increase the likelihood of being absent.
# 
# #### 3. **Interpretability**:
# - **Logistic Regression** is easier to interpret, especially when explaining the relationship between each feature and absenteeism. You can use **odds ratios** to convey how a change in a feature affects the probability of absenteeism.
# - **Random Forest** is harder to interpret, but it's more powerful in capturing **complex patterns** in the data. Feature importance gives a clearer understanding of the key drivers, but it doesn’t directly show the relationships between individual features and the target.
# 
# ---
# 
# ### 📝 **Conclusion**:
# 
# - **Random Forest** is the better choice if you're looking for a model that **captures complex relationships** and **provides powerful insights** about the most important features (like **seasonality** and **transportation costs**).
# - **Logistic Regression**, on the other hand, is valuable for its **simplicity** and **interpretability**, especially when you need to explain **exact feature effects** (using **odds ratios**).
#   
#   However, Logistic Regression **underperforms** when the data has **non-linear patterns** or complex interactions between features, which is why **Random Forest** performed better in terms of accuracy.
# 
# ---
# 
