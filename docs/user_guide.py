"""
AI Lab Super - Comprehensive User Guide
Created by: Mohammad Saeed Angiz
"""

import streamlit as st

st.markdown("# 📖 AI Lab Super - User Guide")
st.markdown("**Created by: Mohammad Saeed Angiz**")
st.markdown("---")

# Table of Contents
st.markdown("""
## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Data Hub](#data-hub)
4. [ML Lab](#ml-lab)
5. [DL Studio](#dl-studio)
6. [Evaluation](#evaluation)
7. [Prediction Hub](#prediction-hub)
8. [Project Management](#project-management)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

Welcome to **AI Lab Super**, an enterprise-grade AI/ML/DL development platform created by **Mohammad Saeed Angiz**. This comprehensive platform provides:

- **Complete ML Workflow**: From data ingestion to model deployment
- **Multiple Frameworks**: Support for scikit-learn, XGBoost, LightGBM, CatBoost, TensorFlow, PyTorch
- **AutoML Integration**: Automated machine learning with FLAML and MLJAR
- **Experiment Tracking**: MLflow integration for tracking all experiments
- **Model Interpretability**: SHAP values for understanding model decisions
- **Interactive Visualizations**: Plotly-based charts and dashboards

### Key Features

✅ **Data Management**
- Upload CSV, Excel, JSON, Parquet files
- Generate synthetic datasets
- Automatic data preprocessing

✅ **Machine Learning**
- 10+ classical ML algorithms
- Automated hyperparameter tuning
- Cross-validation
- Ensemble methods

✅ **Deep Learning**
- Neural network design
- Transfer learning
- Real-time training monitoring
- Model export

✅ **Evaluation & Interpretability**
- Comprehensive metrics
- SHAP explanations
- Model comparison
- Exportable reports

---

## 2. Getting Started

### Step-by-Step Workflow

#### Step 1: Load Your Data

1. Navigate to **Data Hub** from the sidebar
2. Choose one of the following:
   - **Upload Data**: Upload your own dataset (CSV, Excel, JSON, Parquet)
   - **Generate Data**: Create synthetic datasets for testing
   - **Sample Datasets**: Load built-in datasets (Iris, Wine, Breast Cancer)

#### Step 2: Explore Your Data

1. Use **Data Preview** tab to examine:
   - Basic statistics
   - Data types
   - Missing values
   - Duplicates

2. Use **Visualization** tab to:
   - View correlation heatmaps
   - Analyze feature distributions
   - Explore pairwise relationships

#### Step 3: Preprocess Data

1. Handle missing values
2. Scale features
3. Encode categorical variables
4. Perform feature engineering

#### Step 4: Train Models

**Option A: Quick Training**
1. Go to **ML Lab**
2. Select target and feature columns
3. Choose a model
4. Click "Train Model"

**Option B: AutoML**
1. Go to **ML Lab** > **AutoML** tab
2. Configure time budget
3. Let the system find the best model

**Option C: Deep Learning**
1. Go to **DL Studio**
2. Design your neural network
3. Configure training parameters
4. Train the model

#### Step 5: Evaluate Models

1. Navigate to **Evaluation**
2. View performance metrics
3. Analyze SHAP explanations
4. Compare multiple models

#### Step 6: Make Predictions

1. Go to **Prediction Hub**
2. Load your trained model
3. Make single or batch predictions
4. Export results

#### Step 7: Save Your Project

1. Use **Project Management**
2. Save project with all models
3. Export models for deployment

---

## 3. Data Hub

### Upload Data

**Supported Formats:**
- CSV (.csv)
- Excel (.xlsx, .xls)
- JSON (.json)
- Parquet (.parquet)

**Tips:**
- Ensure your data has headers
- Check for encoding issues (UTF-8 recommended)
- Large files may take time to load

### Generate Synthetic Data

**Classification Data:**
- `n_samples`: Number of rows
- `n_features`: Number of columns
- `n_informative`: Number of predictive features
- `n_classes`: Number of target classes

**Regression Data:**
- Similar to classification
- Includes `noise` parameter for realism

### Preprocessing Options

**Missing Value Strategies:**
- Drop rows with missing values
- Drop columns with missing values
- Fill with mean/median/mode
- Fill with constant value

**Feature Scaling:**
- StandardScaler (zero mean, unit variance)
- MinMaxScaler (scale to [0, 1] range)

**Categorical Encoding:**
- Label Encoding (convert to integers)
- One-Hot Encoding (create binary columns)

---

## 4. ML Lab

### Model Training

**Available Models:**

**Classification:**
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Support Vector Machine
5. K-Nearest Neighbors

**Regression:**
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Random Forest Regressor
5. Gradient Boosting Regressor
6. Support Vector Regression
7. K-Nearest Neighbors Regressor

### Hyperparameter Tuning

**Steps:**
1. Select model to tune
2. Choose parameter grid
3. Set cross-validation folds
4. Start search
5. View best parameters

**Tips:**
- Start with small parameter grids
- Increase folds for more reliable results
- Use appropriate scoring metric

### Cross-Validation

**Purpose:**
- Assess model generalization
- Detect overfitting
- Get reliable performance estimates

**Recommended Folds:**
- 5-fold for quick testing
- 10-fold for final evaluation

---

## 5. DL Studio

### Neural Network Design

**Layer Types:**
- Dense (Fully Connected)
- Dropout (Regularization)
- Batch Normalization
- Convolutional (for images)
- LSTM (for sequences)

**Architecture Tips:**
- Start with simple architectures
- Use dropout to prevent overfitting
- Add Batch Normalization for stability
- Use appropriate activation functions

### Training Configuration

**Key Parameters:**
- `epochs`: Number of training iterations
- `batch_size`: Samples per gradient update
- `learning_rate`: Step size for optimization
- `optimizer`: Algorithm for optimization

**Recommended Values:**
- Epochs: 50-200
- Batch Size: 32, 64, or 128
- Learning Rate: 0.001 (Adam default)
- Optimizer: Adam for most cases

### Monitoring Training

**Watch for:**
- Loss decreasing smoothly
- Validation loss not increasing (overfitting)
- Training and validation accuracy converging

**Early Stopping:**
- Stops training when validation loss stops improving
- Prevents overfitting
- Saves best model

---

## 6. Evaluation

### Classification Metrics

- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

### Regression Metrics

- **R² Score**: Proportion of variance explained
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error

### SHAP Explanations

**What is SHAP?**
- SHapley Additive exPlanations
- Game-theoretic approach to explain predictions
- Shows feature contribution to each prediction

**How to Interpret:**
- Positive SHAP value: Feature increases prediction
- Negative SHAP value: Feature decreases prediction
- Magnitude: Importance of feature

---

## 7. Prediction Hub

### Single Prediction

1. Load trained model
2. Enter feature values
3. Get prediction with confidence

### Batch Prediction

1. Upload CSV file with features
2. Model processes all rows
3. Download results

### Model Deployment

**Export Formats:**
- Pickle (.pkl) - Python-specific
- ONNX (.onnx) - Cross-platform
- TensorFlow SavedModel
- PyTorch model

---

## 8. Project Management

### Save Project

**What gets saved:**
- Dataset
- Trained models
- Preprocessing pipelines
- Experiment results

### Load Project

1. Navigate to Project Management
2. Click "Load Project"
3. Select saved project file
4. All models and data restored

### Model Registry

**Features:**
- Version control for models
- Track model performance
- Compare versions
- Promote to production

---

## 9. Advanced Features

### AutoML

**FLAML Integration:**
- Automatic model selection
- Hyperparameter optimization
- Time-budget aware

**Configuration:**
- Set time budget (seconds)
- Choose metric to optimize
- Enable/disable specific models

### MLflow Tracking

**What's Tracked:**
- Parameters
- Metrics
- Model artifacts
- Training code version

**Access MLflow UI:**
```bash
mlflow ui
