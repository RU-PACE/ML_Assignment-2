"""
Machine Learning Assignment 2
Interactive Streamlit Web Application
Adult Income Classification with 6 ML Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Assignment 2 - Adult Income Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #262730;
        padding: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_name):
    """Load trained model from disk"""
    model_files = {
        'Logistic Regression': 'model/logistic_regression_model.pkl',
        'Decision Tree': 'model/decision_tree_model.pkl',
        'kNN': 'model/knn_model.pkl',
        'Naive Bayes': 'model/naive_bayes_model.pkl',
        'Random Forest': 'model/random_forest_model.pkl',
        'XGBoost': 'model/xgboost_model.pkl'
    }
    
    try:
        with open(model_files[model_name], 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found: {model_files[model_name]}")
        return None

@st.cache_resource
def load_scaler():
    """Load the StandardScaler used during training"""
    try:
        with open('model/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.warning("Scaler not found. Using default preprocessing.")
        return None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }
    
    # AUC Score
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

def plot_confusion_matrix(y_true, y_pred):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    return fig

def display_metrics(metrics):
    """Display metrics in a nice layout"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        st.metric("AUC Score", f"{metrics['AUC']:.4f}")
    
    with col2:
        st.metric("Precision", f"{metrics['Precision']:.4f}")
        st.metric("Recall", f"{metrics['Recall']:.4f}")
    
    with col3:
        st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        st.metric("MCC", f"{metrics['MCC']:.4f}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">💰 Adult Income Classification</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 18px; color: #666;">Machine Learning Assignment 2 - Interactive Model Comparison</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    st.sidebar.markdown("### Model Selection")
    
    # Model selection dropdown
    model_options = [
        'Logistic Regression',
        'Decision Tree',
        'kNN',
        'Naive Bayes',
        'Random Forest',
        'XGBoost'
    ]
    
    selected_model = st.sidebar.selectbox(
        "Choose a classification model:",
        model_options,
        help="Select which model to use for predictions"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Information")
    st.sidebar.info("""
    **Dataset**: Adult Income Classification\n
    **Features**: 14 features\n
    **Target**: Binary (Income >50K / <=50K)\n
    **Source**: UCI ML Repository
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    This app demonstrates 6 different machine learning classification models:
    - Logistic Regression
    - Decision Tree
    - K-Nearest Neighbors
    - Naive Bayes
    - Random Forest (Ensemble)
    - XGBoost (Ensemble)
    """)
    
    # Main content
    st.markdown('<div class="sub-header">📊 Dataset Upload</div>', unsafe_allow_html=True)
    
    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your test dataset (CSV format)",
        type=['csv'],
        help="Upload a CSV file with the same features as the training data"
    )
    
    # Option to use demo data
    use_demo_data = st.checkbox("Use demo test data", value=True)
    
    if use_demo_data or uploaded_file is not None:
        
        # Load data
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return
        else:
            # Load demo data
            try:
                df = pd.read_csv('data/test_data.csv')
                st.info("ℹ️ Using demo test data")
            except FileNotFoundError:
                st.error("Demo data not found. Please upload a CSV file.")
                return
        
        # Display dataset info
        st.markdown("### Dataset Preview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1] - 1 if 'target' in df.columns else df.shape[1])
        with col3:
            if 'target' in df.columns:
                st.metric("Classes", df['target'].nunique())
        
        st.dataframe(df.head(10), use_container_width=True)
        
        # Check if target column exists
        if 'target' not in df.columns:
            st.warning("⚠️ 'target' column not found in uploaded data. Using all columns as features for prediction only.")
            X = df
            y = None
        else:
            X = df.drop('target', axis=1)
            y = df['target']
        
        st.markdown("---")
        
        # Model evaluation section
        st.markdown(f'<div class="sub-header">🤖 Model: {selected_model}</div>', unsafe_allow_html=True)
        
        # Load model and scaler
        model = load_model(selected_model)
        scaler = load_scaler()
        
        if model is None:
            st.error("❌ Failed to load model. Please ensure model files are in the 'model/' directory.")
            return
        
        # Make predictions
        try:
            # Scale features if scaler is available
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values
            
            # Get predictions
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)
            
            # Show prediction button
            if st.button("🔮 Run Predictions", type="primary"):
                
                if y is not None:
                    # Calculate metrics
                    st.markdown("### 📈 Evaluation Metrics")
                    metrics = calculate_metrics(y, y_pred, y_pred_proba)
                    display_metrics(metrics)
                    
                    st.markdown("---")
                    
                    # Confusion Matrix
                    st.markdown("### 🎯 Confusion Matrix")
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        fig = plot_confusion_matrix(y, y_pred)
                        st.pyplot(fig)
                    
                    with col2:
                        st.markdown("### 📋 Classification Report")
                        report = classification_report(y, y_pred, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']))
                    
                    st.markdown("---")
                    
                    # Prediction distribution
                    st.markdown("### 📊 Prediction Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        pd.Series(y).value_counts().plot(kind='bar', ax=ax, color='steelblue')
                        ax.set_title('Actual Distribution')
                        ax.set_xlabel('Class')
                        ax.set_ylabel('Count')
                        st.pyplot(fig)
                    
                    with col2:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        pd.Series(y_pred).value_counts().plot(kind='bar', ax=ax, color='coral')
                        ax.set_title('Predicted Distribution')
                        ax.set_xlabel('Class')
                        ax.set_ylabel('Count')
                        st.pyplot(fig)
                
                else:
                    # Just show predictions
                    st.markdown("### 🔮 Predictions")
                    predictions_df = pd.DataFrame({
                        'Prediction': y_pred,
                        'Probability Class 0': y_pred_proba[:, 0],
                        'Probability Class 1': y_pred_proba[:, 1]
                    })
                    st.dataframe(predictions_df.head(20), use_container_width=True)
                    
                    # Download predictions
                    csv = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name=f"{selected_model}_predictions.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.exception(e)
    
    else:
        st.info("👆 Please upload a CSV file or check 'Use demo test data' to begin")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><b>Machine Learning Assignment 2</b> | Adult Income Classification</p>
        <p>Implemented Models: Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
