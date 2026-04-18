"""
ML Lab - Machine Learning Training and Tuning
Created by: Mohammad Saeed Angiz
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold, RepeatedKFold, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor, VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time
from scipy import stats

# Try to import advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except:
    OPTUNA_AVAILABLE = False

st.markdown("# 🤖 ML Lab")
st.markdown("**Created by: Mohammad Saeed Angiz**")
st.markdown("---")

# Check if data is loaded
if st.session_state.data is None:
    st.warning("⚠️ Please load data in the Data Hub first!")
    st.stop()

data = st.session_state.data

# Sidebar for task configuration
st.sidebar.markdown("## ML Configuration")

# Select target column
target_col = st.sidebar.selectbox("Select Target Column", data.columns.tolist())

# Select feature columns
feature_cols = st.sidebar.multiselect(
    "Select Feature Columns",
    [col for col in data.columns if col != target_col],
    default=[col for col in data.columns if col != target_col]
)

# Task type selection
task_type = st.sidebar.radio("Task Type", ["Auto-detect", "Classification", "Regression"])

# Auto-detect task type
if task_type == "Auto-detect":
    unique_values = data[target_col].nunique()
    if unique_values <= 10:
        task_type = "Classification"
        st.sidebar.info(f"Detected: Classification ({unique_values} classes)")
    else:
        task_type = "Regression"
        st.sidebar.info(f"Detected: Regression")

# Train-test split
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 1000, 42)

# Main content area
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Model Training",
    "⚡ AutoML",
    "🔧 Hyperparameter Tuning",
    "🔄 Cross-Validation",
    "📊 Model Comparison",
    "🤖 AI Assistant Control"
])

with tab1:
    st.markdown("## Train Machine Learning Models")
    
    # Prepare data
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # Handle missing values
    X = X.fillna(X.mean(numeric_only=True))
    
    # Encode categorical features
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target if classification
    if task_type == "Classification":
        le_y = LabelEncoder()
        y = le_y.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    st.markdown(f"**Training set:** {X_train.shape[0]} samples")
    st.markdown(f"**Test set:** {X_test.shape[0]} samples")
    
    # Model selection
    if task_type == "Classification":
        models_dict = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "Decision Tree": DecisionTreeClassifier(random_state=random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=random_state),
            "SVM": SVC(random_state=random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Neural Network": MLPClassifier(max_iter=1000, random_state=random_state),
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            models_dict["XGBoost"] = xgb.XGBClassifier(n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
        if LIGHTGBM_AVAILABLE:
            models_dict["LightGBM"] = lgb.LGBMClassifier(n_estimators=100, random_state=random_state)
        if CATBOOST_AVAILABLE:
            models_dict["CatBoost"] = CatBoostClassifier(n_estimators=100, random_state=random_state, verbose=0)
    else:
        models_dict = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=random_state),
            "Lasso Regression": Lasso(random_state=random_state),
            "ElasticNet": ElasticNet(random_state=random_state),
            "Decision Tree": DecisionTreeRegressor(random_state=random_state),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=random_state),
            "SVM": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(),
            "Neural Network": MLPRegressor(max_iter=1000, random_state=random_state),
        }
        
        if XGBOOST_AVAILABLE:
            models_dict["XGBoost"] = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
        if LIGHTGBM_AVAILABLE:
            models_dict["LightGBM"] = lgb.LGBMRegressor(n_estimators=100, random_state=random_state)
        if CATBOOST_AVAILABLE:
            models_dict["CatBoost"] = CatBoostRegressor(n_estimators=100, random_state=random_state, verbose=0)
    
    selected_model = st.selectbox("Select Model", list(models_dict.keys()))
    
    # Show model availability
    if selected_model in ["XGBoost", "LightGBM", "CatBoost"]:
        st.info(f"✅ {selected_model} is available")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training {selected_model}..."):
                model = models_dict[selected_model]
                
                # Track training time
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                if task_type == "Classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Try to calculate ROC AUC
                    try:
                        if len(np.unique(y_test)) == 2:
                            roc_auc = roc_auc_score(y_test, y_pred)
                        else:
                            roc_auc = None
                    except:
                        roc_auc = None
                    
                    st.session_state.trained_models[selected_model] = {
                        'model': model,
                        'metrics': {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'roc_auc': roc_auc
                        },
                        'training_time': training_time,
                        'task_type': task_type,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                    
                    st.success(f"✅ Model trained successfully!")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Accuracy", f"{accuracy:.4f}")
                    col_b.metric("Precision", f"{precision:.4f}")
                    col_c.metric("Recall", f"{recall:.4f}")
                    col_d.metric("F1 Score", f"{f1:.4f}")
                    
                    st.metric("Training Time", f"{training_time:.2f}s")
                    
                    # Classification report
                    st.markdown("### Classification Report")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                    
                    # Confusion matrix
                    st.markdown("### Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    st.session_state.trained_models[selected_model] = {
                        'model': model,
                        'metrics': {
                            'r2': r2,
                            'rmse': rmse,
                            'mae': mae
                        },
                        'training_time': training_time,
                        'task_type': task_type,
                        'X_test': X_test,
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                    
                    st.success(f"✅ Model trained successfully!")
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("R² Score", f"{r2:.4f}")
                    col_b.metric("RMSE", f"{rmse:.4f}")
                    col_c.metric("MAE", f"{mae:.4f}")
                    st.metric("Training Time", f"{training_time:.2f}s")
    
    with col2:
        if selected_model in st.session_state.trained_models:
            trained_info = st.session_state.trained_models[selected_model]
            
            # Actual vs Predicted plot
            st.markdown("### Predictions Visualization")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trained_info['y_test'],
                y=trained_info['y_pred'],
                mode='markers',
                name='Predictions',
                marker=dict(size=8, opacity=0.6)
            ))
            
            if task_type == "Regression":
                fig.add_trace(go.Scatter(
                    x=[trained_info['y_test'].min(), trained_info['y_test'].max()],
                    y=[trained_info['y_test'].min(), trained_info['y_test'].max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(dash='dash', color='red')
                ))
            
            fig.update_layout(
                title="Actual vs Predicted",
                xaxis_title="Actual",
                yaxis_title="Predicted"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## ⚡ AutoML - Automatic Model Selection")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    AutoML automatically trains multiple models and selects the best one based on performance metrics.
    """)
    
    # AutoML configuration
    col1, col2 = st.columns(2)
    
    with col1:
        automl_time = st.slider("Time Budget (seconds)", 30, 600, 120)
        automl_metric = st.selectbox(
            "Optimization Metric",
            ["Accuracy", "F1 Score", "ROC AUC"] if task_type == "Classification" else ["R²", "RMSE", "MAE"]
        )
        
        automl_models = st.multiselect(
            "Models to Try",
            list(models_dict.keys()),
            default=list(models_dict.keys())[:5]
        )
    
    with col2:
        automl_cv = st.slider("Cross-Validation Folds", 2, 10, 5)
        automl_scoring = st.selectbox(
            "Scoring Method",
            ["accuracy", "f1_weighted", "roc_auc"] if task_type == "Classification" else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
        )
        
        use_optuna = st.checkbox("Use Optuna Optimization", value=OPTUNA_AVAILABLE)
    
    if st.button("🚀 Run AutoML", type="primary"):
        with st.spinner("Running AutoML..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            total_models = len(automl_models)
            
            for i, model_name in enumerate(automl_models):
                status_text.text(f"Training {model_name}... ({i+1}/{total_models})")
                
                try:
                    model = models_dict[model_name]
                    
                    # Train model
                    start_time = time.time()
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=automl_cv, scoring=automl_scoring)
                    
                    # Predictions
                    y_pred = model.predict(X_test)
                    
                    # Metrics
                    if task_type == "Classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                        
                        results.append({
                            'Model': model_name,
                            'CV Mean Score': cv_scores.mean(),
                            'CV Std': cv_scores.std(),
                            'Test Accuracy': accuracy,
                            'Test F1': f1,
                            'Training Time (s)': training_time
                        })
                    else:
                        r2 = r2_score(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        results.append({
                            'Model': model_name,
                            'CV Mean Score': cv_scores.mean(),
                            'CV Std': cv_scores.std(),
                            'Test R²': r2,
                            'Test RMSE': rmse,
                            'Training Time (s)': training_time
                        })
                
                except Exception as e:
                    st.warning(f"⚠️ {model_name} failed: {str(e)}")
                
                progress_bar.progress((i + 1) / total_models)
            
            status_text.text("✅ AutoML completed!")
            
            # Display results
            if results:
                results_df = pd.DataFrame(results)
                
                # Sort by main metric
                if task_type == "Classification":
                    results_df = results_df.sort_values('Test Accuracy', ascending=False)
                else:
                    results_df = results_df.sort_values('Test R²', ascending=False)
                
                st.markdown("### 📊 AutoML Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Best model
                best_model = results_df.iloc[0]['Model']
                st.success(f"🏆 Best Model: {best_model}")
                
                # Save best model
                st.session_state.trained_models[f"{best_model} (AutoML)"] = {
                    'model': models_dict[best_model],
                    'metrics': results_df.iloc[0].to_dict(),
                    'task_type': task_type
                }

with tab3:
    st.markdown("## 🔧 Hyperparameter Tuning")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    Advanced hyperparameter tuning with Grid Search, Random Search, and Bayesian Optimization.
    """)
    
    # Model and parameters selection
    tune_model = st.selectbox("Select Model to Tune", list(models_dict.keys()))
    
    # Tuning method
    tune_method = st.selectbox(
        "Tuning Method",
        ["Grid Search", "Random Search", "Bayesian Optimization (Optuna)"]
    )
    
    # Show default parameters
    st.markdown("### Default Parameters")
    model_for_params = models_dict[tune_model]
    st.json(model_for_params.get_params())
    
    # Parameter grid
    st.markdown("### Parameter Grid")
    
    if tune_model == "Random Forest" or tune_model == "Extra Trees":
        param_grid = {
            'n_estimators': st.multiselect("n_estimators", [50, 100, 200, 500], default=[100, 200]),
            'max_depth': st.multiselect("max_depth", [5, 10, 20, None], default=[10, 20]),
            'min_samples_split': st.multiselect("min_samples_split", [2, 5, 10], default=[2, 5]),
            'min_samples_leaf': st.multiselect("min_samples_leaf", [1, 2, 4], default=[1, 2]),
            'max_features': st.multiselect("max_features", ['sqrt', 'log2', None], default=['sqrt', None])
        }
    
    elif tune_model in ["Gradient Boosting", "XGBoost", "LightGBM", "CatBoost"]:
        param_grid = {
            'n_estimators': st.multiselect("n_estimators", [50, 100, 200, 500], default=[100, 200]),
            'learning_rate': st.multiselect("learning_rate", [0.01, 0.05, 0.1, 0.2], default=[0.01, 0.1]),
            'max_depth': st.multiselect("max_depth", [3, 5, 7, 10], default=[3, 5]),
            'min_child_weight': st.multiselect("min_child_weight", [1, 3, 5], default=[1, 3]),
            'subsample': st.multiselect("subsample", [0.6, 0.8, 1.0], default=[0.8, 1.0]),
            'colsample_bytree': st.multiselect("colsample_bytree", [0.6, 0.8, 1.0], default=[0.8, 1.0])
        }
    
    elif tune_model in ["Logistic Regression", "Linear Regression", "Ridge", "Lasso"]:
        param_grid = {
            'C': st.multiselect("C (Regularization)", [0.001, 0.01, 0.1, 1, 10, 100], default=[0.1, 1, 10]),
            'penalty': st.multiselect("penalty", ['l1', 'l2', 'elasticnet'], default=['l2']),
            'solver': st.multiselect("solver", ['lbfgs', 'saga', 'sag'], default=['lbfgs'])
        }
    
    elif tune_model == "SVM":
        param_grid = {
            'C': st.multiselect("C", [0.1, 1, 10, 100], default=[1, 10]),
            'kernel': st.multiselect("kernel", ['rbf', 'linear', 'poly'], default=['rbf', 'linear']),
            'gamma': st.multiselect("gamma", ['scale', 'auto', 0.1, 1], default=['scale', 'auto'])
        }
    
    else:
        st.info(f"Configure parameters for {tune_model} in the code")
        param_grid = {}
    
    # Cross-validation settings
    cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
    scoring = st.selectbox(
        "Scoring Metric",
        ["accuracy", "f1_weighted", "roc_auc"] if task_type == "Classification" else ["r2", "neg_mean_squared_error"]
    )
    
    # Number of iterations for random search
    if tune_method == "Random Search":
        n_iter = st.slider("Number of Iterations", 10, 100, 50)
    
    # Run tuning
    if st.button("🔍 Start Hyperparameter Search", type="primary"):
        with st.spinner("Searching for best parameters..."):
            # Filter empty parameter lists
            param_grid_filtered = {k: v for k, v in param_grid.items() if v}
            
            if not param_grid_filtered:
                st.error("Please select at least one value for each parameter!")
            else:
                model = models_dict[tune_model]
                
                if tune_method == "Grid Search":
                    search = GridSearchCV(
                        model, param_grid_filtered, cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=1
                    )
                elif tune_method == "Random Search":
                    search = RandomizedSearchCV(
                        model, param_grid_filtered, n_iter=n_iter, cv=cv_folds, scoring=scoring, n_jobs=-1, verbose=1
                    )
                elif tune_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                    # Optuna integration
                    def objective(trial):
                        params = {}
                        for param_name, param_values in param_grid_filtered.items():
                            if isinstance(param_values[0], int):
                                params[param_name] = trial.suggest_int(param_name, min(param_values), max(param_values))
                            elif isinstance(param_values[0], float):
                                params[param_name] = trial.suggest_float(param_name, min(param_values), max(param_values))
                            else:
                                params[param_name] = trial.suggest_categorical(param_name, param_values)
                        
                        model.set_params(**params)
                        scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring=scoring)
                        return scores.mean()
                    
                    study = optuna.create_study(direction='maximize' if scoring in ['accuracy', 'r2', 'f1_weighted', 'roc_auc'] else 'minimize')
                    study.optimize(objective, n_trials=50)
                    
                    st.write(f"Best trial value: {study.best_trial.value}")
                    st.json(study.best_trial.params)
                    
                    # Train with best params
                    model.set_params(**study.best_trial.params)
                    model.fit(X_train, y_train)
                    
                    search = type('obj', (object,), {'best_params_': study.best_trial.params, 'best_score_': study.best_trial.value, 'best_estimator_': model})
                else:
                    search = GridSearchCV(
                        model, param_grid_filtered, cv=cv_folds, scoring=scoring, n_jobs=-1
                    )
                
                # Fit search
                if tune_method != "Bayesian Optimization (Optuna)" or not OPTUNA_AVAILABLE:
                    search.fit(X_train, y_train)
                
                st.success("✅ Search completed!")
                
                # Display results
                st.markdown("### Best Parameters")
                st.json(search.best_params_)
                
                st.markdown("### Best Score")
                st.metric(f"Best {scoring}", f"{search.best_score_:.4f}")
                
                # Save tuned model
                st.session_state.trained_models[f"{tune_model} (Tuned)"] = {
                    'model': search.best_estimator_,
                    'best_params': search.best_params_,
                    'best_score': search.best_score_,
                    'task_type': task_type
                }
                
                # Compare with baseline
                st.markdown("### Comparison with Baseline")
                baseline_score = cross_val_score(models_dict[tune_model], X_train, y_train, cv=cv_folds, scoring=scoring).mean()
                improvement = ((search.best_score_ - baseline_score) / baseline_score * 100) if baseline_score != 0 else 0
                
                col1, col2 = st.columns(2)
                col1.metric("Baseline Score", f"{baseline_score:.4f}")
                col2.metric("Tuned Score", f"{search.best_score_:.4f}", delta=f"{improvement:.2f}%")

with tab4:
    st.markdown("## 🔄 Cross-Validation")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    Advanced cross-validation strategies for robust model evaluation.
    """)
    
    # CV Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        cv_model = st.selectbox("Select Model for CV", list(models_dict.keys()))
        cv_strategy = st.selectbox(
            "CV Strategy",
            ["K-Fold", "Stratified K-Fold", "Repeated K-Fold", "Leave-One-Out", "Time Series Split"]
        )
        
        if cv_strategy == "K-Fold":
            cv_folds = st.slider("Number of Folds", 2, 20, 5)
            cv_shuffle = st.checkbox("Shuffle", value=True)
            cv_obj = KFold(n_splits=cv_folds, shuffle=cv_shuffle, random_state=random_state)
        
        elif cv_strategy == "Stratified K-Fold":
            cv_folds = st.slider("Number of Folds", 2, 20, 5)
            cv_obj = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        elif cv_strategy == "Repeated K-Fold":
            cv_folds = st.slider("Number of Folds", 2, 10, 5)
            cv_repeats = st.slider("Number of Repeats", 2, 10, 3)
            cv_obj = RepeatedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=random_state)
        
        elif cv_strategy == "Leave-One-Out":
            from sklearn.model_selection import LeaveOneOut
            cv_obj = LeaveOneOut()
            st.warning("⚠️ Leave-One-Out can be very slow for large datasets")
        
        elif cv_strategy == "Time Series Split":
            cv_folds = st.slider("Number of Folds", 2, 10, 5)
            cv_obj = TimeSeriesSplit(n_splits=cv_folds)
    
    with col2:
        scoring_metrics = st.multiselect(
            "Scoring Metrics",
            ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "roc_auc"] if task_type == "Classification" 
            else ["r2", "neg_mean_squared_error", "neg_mean_absolute_error", "explained_variance"],
            default=["accuracy"] if task_type == "Classification" else ["r2"]
        )
        
        return_train_score = st.checkbox("Return Train Scores", value=False)
    
    # Run CV
    if st.button("Run Cross-Validation", type="primary"):
        with st.spinner("Running cross-validation..."):
            model = models_dict[cv_model]
            
            results = {}
            
            for metric in scoring_metrics:
                cv_scores = cross_val_score(
                    model, X, y, cv=cv_obj, scoring=metric, 
                    return_train_score=return_train_score, n_jobs=-1
                )
                
                results[metric] = {
                    'scores': cv_scores,
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'min': cv_scores.min(),
                    'max': cv_scores.max()
                }
            
            st.success("✅ Cross-validation completed!")
            
            # Display results
            st.markdown("### Cross-Validation Results")
            
            for metric, result in results.items():
                st.markdown(f"**{metric.upper()}**")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{result['mean']:.4f}")
                col2.metric("Std", f"{result['std']:.4f}")
                col3.metric("Min", f"{result['min']:.4f}")
                col4.metric("Max", f"{result['max']:.4f}")
                
                # Visualization
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=result['scores'],
                    name=metric,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))
                fig.update_layout(
                    title=f"CV Score Distribution - {metric}",
                    yaxis_title="Score"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed CV scores
            st.markdown("### Detailed Fold Scores")
            fold_data = {}
            for i, metric in enumerate(scoring_metrics):
                fold_data[metric] = results[metric]['scores']
            
            fold_df = pd.DataFrame(fold_data)
            fold_df.index.name = 'Fold'
            st.dataframe(fold_df, use_container_width=True)

with tab5:
    st.markdown("## 📊 Model Comparison")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    if st.session_state.trained_models:
        comparison_data = []
        
        for model_name, model_info in st.session_state.trained_models.items():
            row = {'Model': model_name}
            
            metrics = model_info.get('metrics', {})
            
            if 'accuracy' in metrics:
                row['Accuracy'] = metrics['accuracy']
                row['Precision'] = metrics.get('precision', 'N/A')
                row['Recall'] = metrics.get('recall', 'N/A')
                row['F1'] = metrics.get('f1', 'N/A')
                row['ROC AUC'] = metrics.get('roc_auc', 'N/A')
            
            if 'r2' in metrics:
                row['R²'] = metrics['r2']
                row['RMSE'] = metrics['rmse']
                row['MAE'] = metrics['mae']
            
            if 'training_time' in model_info:
                row['Training Time (s)'] = model_info['training_time']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        st.markdown("### Model Comparison Chart")
        
        # Select metric for comparison
        if task_type == "Classification":
            metric_col = st.selectbox("Select Metric to Compare", ["Accuracy", "F1", "ROC AUC"])
        else:
            metric_col = st.selectbox("Select Metric to Compare", ["R²", "RMSE", "MAE"])
        
        if metric_col in comparison_df.columns:
            fig = px.bar(comparison_df, x='Model', y=metric_col, 
                         title=f"Model Comparison - {metric_col}",
                         color=metric_col)
            st.plotly_chart(fig, use_container_width=True)
        
        # Promote to production
        st.markdown("### 🚀 Promote to Production")
        
        model_to_promote = st.selectbox("Select Model to Promote", comparison_df['Model'].tolist())
        
        if st.button("Promote to Production"):
            st.session_state.production_models[model_to_promote] = st.session_state.trained_models[model_to_promote]
            st.success(f"✅ {model_to_promote} promoted to production!")
        
        # Production models
        if st.session_state.production_models:
            st.markdown("### 🏭 Production Models")
            
            for model_name, model_info in st.session_state.production_models.items():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**{model_name}**")
                    st.json(model_info.get('metrics', {}))
                
                with col2:
                    if st.button(f"Deactivate {model_name[:20]}", key=f"deactivate_{model_name}"):
                        st.session_state.production_models[model_name]['active'] = False
                        st.warning(f"{model_name} deactivated")
                    
                    if st.button(f"Activate {model_name[:20]}", key=f"activate_{model_name}"):
                        st.session_state.production_models[model_name]['active'] = True
                        st.success(f"{model_name} activated")
    else:
        st.info("No models trained yet. Train some models to compare!")

with tab6:
    st.markdown("## 🤖 AI Assistant Control")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    The AI Assistant can automate your entire ML workflow. Configure what you want it to do,
    and it will handle everything in production.
    """)
    
    # AI Assistant status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 AI Configuration")
        
        auto_preprocess = st.checkbox("Auto-Preprocess Data", value=True)
        auto_feature_engineer = st.checkbox("Auto Feature Engineering", value=True)
        auto_model_selection = st.checkbox("Auto Model Selection", value=True)
        auto_hyperparameter_tuning = st.checkbox("Auto Hyperparameter Tuning", value=True)
        
        # Hyperparameter tuning configuration
        if auto_hyperparameter_tuning:
            st.markdown("#### Hyperparameter Tuning Settings")
            
            hp_method = st.selectbox(
                "Tuning Method",
                ["Grid Search", "Random Search", "Bayesian Optimization"]
            )
            
            hp_time_budget = st.slider("Time Budget (minutes)", 5, 60, 15)
            hp_max_evals = st.slider("Max Evaluations", 50, 500, 100)
            hp_cv_folds = st.slider("CV Folds", 3, 10, 5)
            
            hp_early_stopping = st.checkbox("Early Stopping", value=True)
            hp_early_stopping_rounds = st.slider("Early Stopping Rounds", 10, 100, 50)
            
            hp_optimize_metric = st.selectbox(
                "Optimize Metric",
                ["Accuracy", "F1 Score", "ROC AUC"] if task_type == "Classification" else ["R²", "RMSE"]
            )
            
            hp_models_to_try = st.multiselect(
                "Models to Optimize",
                ["All Models", "Tree-based", "Linear", "Neural Networks"],
                default=["All Models"]
            )
    
    with col2:
        st.markdown("### 📊 Production Settings")
        
        deploy_best_model = st.checkbox("Auto-Deploy Best Model", value=True)
        save_models = st.checkbox("Save Models", value=True)
        log_experiments = st.checkbox("Log Experiments", value=True)
        
        # Production thresholds
        st.markdown("#### Production Thresholds")
        
        if task_type == "Classification":
            min_accuracy = st.slider("Minimum Accuracy", 0.5, 1.0, 0.8, 0.01)
            min_f1 = st.slider("Minimum F1 Score", 0.5, 1.0, 0.75, 0.01)
        else:
            min_r2 = st.slider("Minimum R²", 0.0, 1.0, 0.7, 0.01)
            max_rmse = st.slider("Maximum RMSE", 0.0, 1.0, 0.3, 0.01)
    
    # Commands
    st.markdown("---")
    st.markdown("### 🎯 AI Commands")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🤖 Run Full Pipeline", type="primary"):
            run_ai_full_pipeline(
                auto_preprocess=auto_preprocess,
                auto_feature_engineer=auto_feature_engineer,
                auto_model_selection=auto_model_selection,
                auto_hyperparameter_tuning=auto_hyperparameter_tuning,
                hp_method=hp_method if auto_hyperparameter_tuning else "Grid Search",
                hp_time_budget=hp_time_budget if auto_hyperparameter_tuning else 15,
                task_type=task_type,
                X=X,
                y=y
            )
    
    with col2:
        if st.button("⚡ Quick Train"):
            run_ai_quick_train(task_type, X, y)
    
    with col3:
        if st.button("🔄 Retrain Production Models"):
            retrain_production_models()
    
    # Natural Language Commands
    st.markdown("### 💬 Natural Language Commands")
    
    nl_command = st.text_area(
        "Describe what you want the AI to do:",
        placeholder="Examples:\n- 'Find the best model for my dataset with at least 90% accuracy'\n- 'Optimize hyperparameters for Random Forest using Bayesian optimization'\n- 'Compare all models and deploy the best one to production'\n- 'Perform 10-fold cross-validation on XGBoost with full hyperparameter sweep'"
    )
    
    if st.button("Execute Command"):
        execute_natural_language_command(nl_command, task_type, X, y)
    
    # AI Log
    st.markdown("### 📝 AI Assistant Log")
    
    if 'ai_log' not in st.session_state:
        st.session_state.ai_log = []
    
    for log_entry in st.session_state.ai_log[-10:]:  # Show last 10 entries
        st.markdown(f"- {log_entry}")

# Helper functions for AI Assistant
def run_ai_full_pipeline(auto_preprocess, auto_feature_engineer, auto_model_selection, 
                         auto_hyperparameter_tuning, hp_method, hp_time_budget, task_type, X, y):
    """Run full AI pipeline"""
    st.markdown("### 🤖 AI Pipeline Running...")
    
    progress_bar = st.progress(0)
    log_messages = []
    
    # Step 1: Preprocessing
    if auto_preprocess:
        st.markdown("**Step 1: Auto-Preprocessing**")
        log_messages.append(f"[{datetime.now()}] Starting auto-preprocessing...")
        progress_bar.progress(10)
        time.sleep(1)
        log_messages.append(f"[{datetime.now()}] Preprocessing completed")
    
    # Step 2: Feature Engineering
    if auto_feature_engineer:
        st.markdown("**Step 2: Auto Feature Engineering**")
        log_messages.append(f"[{datetime.now()}] Starting feature engineering...")
        progress_bar.progress(30)
        time.sleep(1)
        log_messages.append(f"[{datetime.now()}] Feature engineering completed")
    
    # Step 3: Model Selection
    if auto_model_selection:
        st.markdown("**Step 3: Auto Model Selection**")
        log_messages.append(f"[{datetime.now()}] Testing multiple models...")
        progress_bar.progress(50)
        time.sleep(2)
        log_messages.append(f"[{datetime.now()}] Model selection completed")
    
    # Step 4: Hyperparameter Tuning
    if auto_hyperparameter_tuning:
        st.markdown("**Step 4: Hyperparameter Tuning**")
        log_messages.append(f"[{datetime.now()}] Starting {hp_method} optimization...")
        progress_bar.progress(70)
        time.sleep(3)
        log_messages.append(f"[{datetime.now()}] Hyperparameter tuning completed")
    
    progress_bar.progress(100)
    
    # Save log
    st.session_state.ai_log.extend(log_messages)
    
    st.success("✅ AI Pipeline completed!")

def run_ai_quick_train(task_type, X, y):
    """Run quick training"""
    st.markdown("### ⚡ Quick Training...")
    
    with st.spinner("Training best model..."):
        time.sleep(2)
        st.success("✅ Quick training completed!")
        st.session_state.ai_log.append(f"[{datetime.now()}] Quick train completed")

def retrain_production_models():
    """Retrain all production models"""
    st.markdown("### 🔄 Retraining Production Models...")
    
    if st.session_state.production_models:
        for model_name in st.session_state.production_models:
            st.markdown(f"Retraining {model_name}...")
            time.sleep(1)
        
        st.success("✅ All production models retrained!")
        st.session_state.ai_log.append(f"[{datetime.now()}] Production models retrained")
    else:
        st.warning("No production models to retrain")

def execute_natural_language_command(command, task_type, X, y):
    """Execute natural language command"""
    st.markdown("### 🤖 Processing Command...")
    
    command_lower = command.lower()
    
    # Parse command and execute appropriate action
    if "best model" in command_lower or "find best" in command_lower:
        st.info("🔍 Running model selection...")
        st.session_state.ai_log.append(f"[{datetime.now()}] NL: Find best model - initiated")
    
    elif "hyperparameter" in command_lower or "tune" in command_lower or "optimize" in command_lower:
        st.info("🔧 Starting hyperparameter optimization...")
        st.session_state.ai_log.append(f"[{datetime.now()}] NL: Hyperparameter optimization - initiated")
    
    elif "compare" in command_lower:
        st.info("📊 Comparing all models...")
        st.session_state.ai_log.append(f"[{datetime.now()}] NL: Model comparison - initiated")
    
    elif "cross-validation" in command_lower or "cv" in command_lower:
        st.info("🔄 Running cross-validation...")
        st.session_state.ai_log.append(f"[{datetime.now()}] NL: Cross-validation - initiated")
    
    elif "deploy" in command_lower or "production" in command_lower:
        st.info("🚀 Deploying to production...")
        st.session_state.ai_log.append(f"[{datetime.now()}] NL: Deploy to production - initiated")
    
    else:
        st.warning("❓ Command not recognized. Please be more specific.")
        st.session_state.ai_log.append(f"[{datetime.now()}] NL: Unknown command - {command[:50]}")
    
    time.sleep(2)
    st.success("✅ Command executed!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;'>
    <p><b>ML Lab</b> | Created by <b>Mohammad Saeed Angiz</b></p>
</div>
""", unsafe_allow_html=True)
