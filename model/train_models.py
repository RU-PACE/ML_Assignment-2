"""
Machine Learning Assignment 2
Train and Evaluate 6 Classification Models
Dataset: Heart Disease Classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_adult_income_data():
    """
    Load Adult Income dataset from UCI repository
    Features: 14 (age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, income)
    Instances: 32,561
    """
    # Column names for Adult dataset
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    try:
        # Load training data only (32,561 instances - meets >500 requirement!)
        url_train = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        df = pd.read_csv(url_train, names=column_names, na_values=' ?', skipinitialspace=True)
        print("✓ Dataset loaded from UCI repository")
        
    except Exception as e:
        print(f"Error loading from UCI: {e}")
        print("Creating sample dataset locally...")
        df = create_sample_adult_data()
    
    # Rename target column
    if 'income' in df.columns:
        df = df.rename(columns={'income': 'target'})
    
    # Convert target to binary (0: <=50K, 1: >50K)
    df['target'] = df['target'].astype(str).str.strip()
    df['target'] = df['target'].apply(lambda x: 1 if '>50K' in x else 0)
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    from sklearn.preprocessing import LabelEncoder
    le_dict = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
    
    print("✓ Categorical variables encoded")
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Instances: {len(df)}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    
    return df

def create_sample_adult_data():
    """Fallback function to create sample adult income data"""
    np.random.seed(42)
    n_samples = 10000  # Sufficient for requirements
    
    data = {
        'age': np.random.randint(17, 91, n_samples),
        'workclass': np.random.randint(0, 9, n_samples),
        'fnlwgt': np.random.randint(12285, 1490400, n_samples),
        'education': np.random.randint(0, 16, n_samples),
        'education-num': np.random.randint(1, 17, n_samples),
        'marital-status': np.random.randint(0, 7, n_samples),
        'occupation': np.random.randint(0, 15, n_samples),
        'relationship': np.random.randint(0, 6, n_samples),
        'race': np.random.randint(0, 5, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'capital-gain': np.random.randint(0, 100000, n_samples),
        'capital-loss': np.random.randint(0, 5000, n_samples),
        'hours-per-week': np.random.randint(1, 100, n_samples),
        'native-country': np.random.randint(0, 42, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all required evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }
    
    # AUC Score (requires probability predictions)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:  # Multi-class
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train all 6 models and calculate metrics"""
    
    results = {}
    models = {}
    
    print("\n" + "="*80)
    print("Training and Evaluating Models")
    print("="*80)
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_pred_proba_lr = lr.predict_proba(X_test)
    results['Logistic Regression'] = calculate_all_metrics(y_test, y_pred_lr, y_pred_proba_lr)
    models['Logistic Regression'] = lr
    print("✓ Logistic Regression completed")
    
    # 2. Decision Tree Classifier
    print("\n2. Training Decision Tree Classifier...")
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_pred_proba_dt = dt.predict_proba(X_test)
    results['Decision Tree'] = calculate_all_metrics(y_test, y_pred_dt, y_pred_proba_dt)
    models['Decision Tree'] = dt
    print("✓ Decision Tree completed")
    
    # 3. K-Nearest Neighbors Classifier
    print("\n3. Training K-Nearest Neighbors Classifier...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_pred_proba_knn = knn.predict_proba(X_test)
    results['kNN'] = calculate_all_metrics(y_test, y_pred_knn, y_pred_proba_knn)
    models['kNN'] = knn
    print("✓ K-Nearest Neighbors completed")
    
    # 4. Naive Bayes Classifier (Gaussian)
    print("\n4. Training Naive Bayes Classifier (Gaussian)...")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    y_pred_proba_nb = nb.predict_proba(X_test)
    results['Naive Bayes'] = calculate_all_metrics(y_test, y_pred_nb, y_pred_proba_nb)
    models['Naive Bayes'] = nb
    print("✓ Naive Bayes completed")
    
    # 5. Random Forest (Ensemble)
    print("\n5. Training Random Forest (Ensemble)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_pred_proba_rf = rf.predict_proba(X_test)
    results['Random Forest (Ensemble)'] = calculate_all_metrics(y_test, y_pred_rf, y_pred_proba_rf)
    models['Random Forest'] = rf
    print("✓ Random Forest completed")
    
    # 6. XGBoost (Ensemble)
    print("\n6. Training XGBoost (Ensemble)...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, max_depth=5, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_pred_proba_xgb = xgb.predict_proba(X_test)
    results['XGBoost (Ensemble)'] = calculate_all_metrics(y_test, y_pred_xgb, y_pred_proba_xgb)
    models['XGBoost'] = xgb
    print("✓ XGBoost completed")
    
    return results, models

def print_results_table(results):
    """Print results in a formatted table"""
    print("\n" + "="*100)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*100)
    
    # Header
    print(f"{'ML Model Name':<30} {'Accuracy':<12} {'AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MCC':<12}")
    print("-" * 100)
    
    # Data rows
    for model_name, metrics in results.items():
        print(f"{model_name:<30} "
              f"{metrics['Accuracy']:<12.4f} "
              f"{metrics['AUC']:<12.4f} "
              f"{metrics['Precision']:<12.4f} "
              f"{metrics['Recall']:<12.4f} "
              f"{metrics['F1']:<12.4f} "
              f"{metrics['MCC']:<12.4f}")
    
    print("="*100)

def save_models_and_data(models, scaler, X_test, y_test):
    """Save trained models and test data"""
    print("\n" + "="*80)
    print("Saving Models and Data")
    print("="*80)
    
    # Save each model
    for model_name, model in models.items():
        filename = f"model/{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved {model_name} to {filename}")
    
    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler to model/scaler.pkl")
    
    # Save test data for Streamlit demo
    test_data = pd.DataFrame(X_test)
    test_data['target'] = y_test
    test_data.to_csv('data/test_data.csv', index=False)
    print(f"✓ Saved test data to data/test_data.csv")
    
    print("="*80)

def main():
    """Main training pipeline"""
    print("="*80)
    print("MACHINE LEARNING ASSIGNMENT 2")
    print("Adult Income Classification - 6 Models Comparison")
    print("="*80)
    
    # Load data
    df = load_adult_income_data()
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Save original dataset
    df.to_csv('data/adult_income_full.csv', index=False)
    print(f"✓ Saved full dataset to data/adult_income_full.csv")
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate all models
    results, models = train_and_evaluate_models(
        X_train_scaled, X_test_scaled, y_train, y_test
    )
    
    # Print results table
    print_results_table(results)
    
    # Save everything
    save_models_and_data(models, scaler, X_test, y_test.values)
    
    # Save results to CSV for README
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model/results_summary.csv')
    print(f"\n✓ Results summary saved to model/results_summary.csv")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Check model/ folder for saved models")
    print("2. Check data/ folder for datasets")
    print("3. Run the Streamlit app: streamlit run app.py")
    print("="*80)

if __name__ == "__main__":
    main()
