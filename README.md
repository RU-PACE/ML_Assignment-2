# Adult Income Classification using Multiple ML Models

## 1.  Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict whether an individual's annual income exceeds $50,000 based on demographic and employment features. This project implements six different classification algorithms and evaluates their performance using comprehensive metrics to determine the most effective model for income classification.

The classification problem is binary:
- **Class 0**: Income ≤ $50K per year
- **Class 1**: Income > $50K per year

---

## 2.  Dataset Description

### a. Dataset: Adult Income Dataset (UCI Machine Learning Repository)

**Source**: UCI Machine Learning Repository - Adult Census Income Dataset

**Overview**:
The Adult Income dataset (also known as "Census Income" dataset) contains demographic and employment data extracted from the 1994 Census database. It is widely used for binary classification tasks in machine learning to predict whether an individual earns more than $50,000 annually based on various factors.

**Dataset Characteristics**:
- **Number of Instances**: 32,561 samples in training data
- **Number of Features**: 14 demographic and employment features + 1 target variable
- **Feature Types**: Mix of continuous (6) and categorical (8) variables
- **Class Distribution**: 
  - Class 0 (Income ≤$50K): ~76% (24,720 samples)
  - Class 1 (Income >$50K): ~24% (7,841 samples)
- **Missing Values**: Minimal (handled via encoding)

### b.  Feature Descriptions:

| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| **age** | Age of the individual | Continuous | 17-90 years |
| **workclass** | Employment type | Categorical | Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked |
| **fnlwgt** | Census sampling weight | Continuous | 12,285 - 1,484,705 |
| **education** | Highest education level | Categorical | Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool |
| **education-num** | Education level in numeric form | Continuous | 1-16 |
| **marital-status** | Marital status | Categorical | Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse |
| **occupation** | Type of occupation | Categorical | Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces |
| **relationship** | Relationship status | Categorical | Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried |
| **race** | Race of the individual | Categorical | White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black |
| **sex** | Gender | Binary | Male, Female |
| **capital-gain** | Capital gains | Continuous | 0 - 99,999 |
| **capital-loss** | Capital losses | Continuous | 0 - 4,356 |
| **hours-per-week** | Hours worked per week | Continuous | 1-99 |
| **native-country** | Country of origin | Categorical | United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands |
| **income** | Income class (target variable) | Binary | <=50K, >50K |
---

## 3. Model Performance Comparison Table


| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8279 | 0.8608 | 0.8280 | 0.8279 | 0.8254 | 0.5796 |
| Decision Tree | 0.8543 | 0.8909 | 0.8544 | 0.8543 | 0.8536 | 0.6438 |
| kNN | 0.8340 | 0.8569 | 0.8378 | 0.8340 | 0.8291 | 0.5955 |
| Naive Bayes | 0.8081 | 0.8644 | 0.8092 | 0.8081 | 0.8036 | 0.5340 |
| Random Forest (Ensemble) | 0.8620 | 0.9196 | 0.8621 | 0.8620 | 0.8613 | 0.6582 |
| XGBoost (Ensemble) | 0.8721 | 0.9295 | 0.8722 | 0.8721 | 0.8716 | 0.6829 |

##### Note : These are the actual results from model training on the test set (6,513 samples) from the Adult Income dataset.
---

## 4.  Model Performance Observations Summary

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieved 82.79% accuracy with strong AUC (0.8608). Simple linear model that performs competitively on socioeconomic data. Excellent interpretability and fast training make it ideal for explaining income factors. Good balance between precision and recall despite class imbalance. |
| **Decision Tree** | Strong performance with 85.43% accuracy and AUC of 0.8909. Provides excellent interpretability through visualization of decision rules (education-num > threshold, capital-gain splits). Performance improves when converted to ensemble. |
| **kNN** | Solid accuracy of 83.40% with AUC of 0.8569. Benefits significantly from StandardScaler preprocessing due to varying feature scales. Instance-based learning captures local demographic patterns effectively. Computationally expensive during prediction with 26,048 training samples. K=5 is effective for this dataset. |
| **Naive Bayes** | Acceptable accuracy of 80.81% (lowest among all models) with AUC of 0.8644. Works moderately well despite Gaussian independence assumption. Extremely fast training and prediction times - ideal for real-time estimation. MCC of 0.5340 indicates some difficulty with imbalanced class distribution. 7% performance gap compared to best model. |
| **Random Forest (Ensemble)** | Excellent performance with 86.20% accuracy and strong AUC (0.9196). Reduces overfitting significantly compared to single Decision Tree. Provides valuable feature importance rankings. Robust to outliers and handles categorical encoding effectively. Ensemble of 100 trees provides stable predictions through bootstrap aggregating. |
| **XGBoost (Ensemble)** | **Best performer** with highest accuracy of 87.21% and best AUC (0.9295). Best MCC score (0.6829) indicating superior balanced performance despite class imbalance. Gradient boosting with regularization prevents overfitting on large dataset (32,561 samples). Industry-standard algorithm, handles missing values implicitly, and robust to outliers.

---

## End of README
