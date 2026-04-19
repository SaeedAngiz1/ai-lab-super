import os
import json
import uuid
import pickle
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


APP_TITLE = "AI LAB Advanced"
CONFIG_PATH = "config.json"
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------
# Session Initialization
# --------------------------
def init_session_state() -> None:
    defaults = {
        "data": None,
        "target_column": None,
        "trained_model": None,
        "model_name": None,
        "model_type": None,
        "X_test": None,
        "y_test": None,
        "feature_columns": None,
        "label_encoder": None,
        "preprocessor": None,
        "metrics": None,
        "chat_history": [],
        "dl_layers": [{"units": 64, "activation": "relu", "dropout": 0.0}],
        "dl_model": None,
        "dl_history": None,
        "ai_provider_config": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state.ai_provider_config is None:
        st.session_state.ai_provider_config = default_config()


# --------------------------
# Config Management
# --------------------------
def default_config() -> Dict[str, Any]:
    return {
        "provider": "openai",
        "openai": {
            "api_key": "",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1/chat/completions",
        },
        "anthropic": {
            "api_key": "",
            "model": "claude-3-5-sonnet-20241022",
            "base_url": "https://api.anthropic.com/v1/messages",
        },
        "ollama": {
            "base_url": "http://localhost:11434/api/chat",
            "model": "llama3.1",
        },
        "custom": {
            "api_key": "",
            "base_url": "",
            "model": "",
            "headers": "{}",
            "payload_template": '{"model": "{model}", "messages": [{"role": "user", "content": "{prompt}"}] }',
        },
    }


def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        cfg = default_config()
        save_config(cfg)
        return cfg
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default_config()


def save_config(cfg: Dict[str, Any]) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def get_active_ai_config() -> Dict[str, Any]:
    cfg = st.session_state.get("ai_provider_config")
    if not cfg:
        cfg = default_config()
        st.session_state.ai_provider_config = cfg
    return cfg


def provider_is_configured(provider: str, cfg: Dict[str, Any]) -> bool:
    p = cfg.get(provider, {})
    if provider in {"openai", "anthropic"}:
        return bool(p.get("api_key")) and bool(p.get("model"))
    if provider == "ollama":
        return bool(p.get("base_url")) and bool(p.get("model"))
    if provider == "custom":
        return bool(p.get("base_url")) and bool(p.get("api_key")) and bool(p.get("model"))
    return False


def safe_filename_part(text: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in str(text or "model"))
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "model"


def build_model_download_name(model_file: str, metadata: Dict[str, Any]) -> str:
    model_name = safe_filename_part(metadata.get("model_name") or os.path.splitext(model_file)[0])
    saved_at = str(metadata.get("saved_at_utc", ""))
    year = saved_at[:4] if len(saved_at) >= 4 and saved_at[:4].isdigit() else datetime.utcnow().strftime("%Y")
    return f"{model_name}_model_{year}.pkl"


# --------------------------
# Utility Functions
# --------------------------
def infer_problem_type(y: pd.Series) -> str:
    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    return "regression"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", SimpleImputer(strategy="most_frequent")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ], remainder="drop")


def get_model_and_grid(model_key: str, problem_type: str):
    if model_key == "Random Forest":
        if problem_type == "classification":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            }
        else:
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 5],
            }
        return model, param_grid

    if model_key == "XGBoost":
        if XGBClassifier is None or XGBRegressor is None:
            raise RuntimeError("xgboost is not installed. Please install xgboost.")
        if problem_type == "classification":
            model = XGBClassifier(random_state=42, eval_metric="logloss")
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.1],
            }
        else:
            model = XGBRegressor(random_state=42)
            param_grid = {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.1],
            }
        return model, param_grid

    if model_key == "Logistic Regression":
        if problem_type != "classification":
            raise RuntimeError("Logistic Regression is only available for classification.")
        model = LogisticRegression(max_iter=2000)
        param_grid = {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["lbfgs", "liblinear"],
        }
        return model, param_grid

    raise ValueError("Unsupported model selection")


def evaluate_model(model, X_test, y_test, problem_type: str) -> Dict[str, float]:
    pred = model.predict(X_test)
    result = {}
    if problem_type == "classification":
        result["accuracy"] = accuracy_score(y_test, pred)
        result["precision_weighted"] = precision_score(y_test, pred, average="weighted", zero_division=0)
        result["recall_weighted"] = recall_score(y_test, pred, average="weighted", zero_division=0)
        result["f1_weighted"] = f1_score(y_test, pred, average="weighted", zero_division=0)
    else:
        result["rmse"] = float(np.sqrt(mean_squared_error(y_test, pred)))
        result["mae"] = float(mean_absolute_error(y_test, pred))
        result["r2"] = float(r2_score(y_test, pred))
    return result


def safe_json_load(text: str, default=None):
    try:
        return json.loads(text)
    except Exception:
        return {} if default is None else default


def ai_request(provider: str, cfg: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    timeout = 90
    user_prompt = messages[-1]["content"] if messages else ""

    if provider == "openai":
        p = cfg.get("openai", {})
        headers = {
            "Authorization": f"Bearer {p.get('api_key', '')}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": p.get("model", "gpt-4o-mini"),
            "messages": messages,
            "temperature": 0.3,
        }
        r = requests.post(p.get("base_url", "https://api.openai.com/v1/chat/completions"), headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    if provider == "anthropic":
        p = cfg.get("anthropic", {})
        headers = {
            "x-api-key": p.get("api_key", ""),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        anthropic_messages = [{"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]} for m in messages]
        payload = {
            "model": p.get("model", "claude-3-5-sonnet-20241022"),
            "max_tokens": 1024,
            "messages": anthropic_messages,
        }
        r = requests.post(p.get("base_url", "https://api.anthropic.com/v1/messages"), headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("content", [{}])[0].get("text", "No response")

    if provider == "ollama":
        p = cfg.get("ollama", {})
        payload = {
            "model": p.get("model", "llama3.1"),
            "messages": messages,
            "stream": False,
        }
        r = requests.post(p.get("base_url", "http://localhost:11434/api/chat"), json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "No response")

    if provider == "custom":
        p = cfg.get("custom", {})
        headers = safe_json_load(p.get("headers", "{}"), default={})
        if p.get("api_key"):
            headers.setdefault("Authorization", f"Bearer {p['api_key']}")
        headers.setdefault("Content-Type", "application/json")

        template = p.get("payload_template", "")
        rendered = template.replace("{prompt}", user_prompt.replace('"', "\\\""))
        rendered = rendered.replace("{model}", p.get("model", ""))
        payload = safe_json_load(rendered, default={"prompt": user_prompt})

        r = requests.post(p.get("base_url", ""), headers=headers, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict):
            if "choices" in data and data["choices"]:
                return data["choices"][0].get("message", {}).get("content", str(data))
            if "response" in data:
                return data["response"]
            if "message" in data and isinstance(data["message"], dict):
                return data["message"].get("content", str(data))
        return str(data)

    raise ValueError("Unsupported provider")


# --------------------------
# Page: Data Hub
# --------------------------
def page_data_hub() -> None:
    st.header("📊 Data Hub")
    st.write("Upload CSV/JSON data, preprocess it, and visualize insights.")

    file = st.file_uploader("Upload CSV or JSON", type=["csv", "json"])
    if file is not None:
        try:
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_json(file)
            st.session_state.data = df
            st.success(f"Loaded dataset with shape: {df.shape}")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    if st.session_state.data is None:
        st.info("Please upload a dataset to continue.")
        return

    df = st.session_state.data
    st.subheader("Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    with st.expander("Preprocessing", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Drop Duplicates"):
                before = len(df)
                st.session_state.data = df.drop_duplicates()
                st.success(f"Removed {before - len(st.session_state.data)} duplicate rows.")

            if st.button("Fill Numeric NA with Median"):
                d = st.session_state.data.copy()
                num_cols = d.select_dtypes(include=np.number).columns
                d[num_cols] = d[num_cols].fillna(d[num_cols].median())
                st.session_state.data = d
                st.success("Filled numeric missing values with median.")

        with col2:
            if st.button("Drop Rows with Any NA"):
                before = len(df)
                st.session_state.data = df.dropna()
                st.success(f"Dropped {before - len(st.session_state.data)} rows containing NA values.")

            if st.button("Reset Index"):
                st.session_state.data = df.reset_index(drop=True)
                st.success("Index reset done.")

    df = st.session_state.data

    st.subheader("Visualization")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    chart_type = st.selectbox("Chart Type", ["Histogram", "Box", "Scatter", "Line", "Bar"])

    if chart_type in ["Histogram", "Box", "Bar", "Line"] and all_cols:
        x_col = st.selectbox("X-axis", all_cols, key="viz_x")
        color_col = st.selectbox("Color (optional)", [None] + all_cols, key="viz_color")

        if st.button("Generate Plot"):
            try:
                if chart_type == "Histogram":
                    fig = px.histogram(df, x=x_col, color=color_col)
                elif chart_type == "Box":
                    fig = px.box(df, x=x_col, y=numeric_cols[0] if numeric_cols else None, color=color_col)
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x_col, color=color_col)
                else:
                    fig = px.line(df, x=df.index, y=x_col, color=color_col)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Plot error: {e}")

    if chart_type == "Scatter":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for scatter plot.")
        else:
            x_col = st.selectbox("X", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y", numeric_cols, key="scatter_y")
            color_col = st.selectbox("Color (optional)", [None] + all_cols, key="scatter_color")
            if st.button("Generate Scatter"):
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
                st.plotly_chart(fig, use_container_width=True)


# --------------------------
# Page: ML Training
# --------------------------
def page_ml_training() -> None:
    st.header("🧠 ML Training")
    if st.session_state.data is None:
        st.info("Please upload data first in Data Hub.")
        return

    df = st.session_state.data.copy()
    target = st.selectbox("Select Target Column", df.columns.tolist())
    model_choice = st.selectbox("Model", ["Random Forest", "XGBoost", "Logistic Regression"])
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    cv_folds = st.slider("K-Fold CV", 3, 10, 5)

    if st.button("Train Model"):
        progress = st.progress(0)
        status = st.empty()

        try:
            status.info("Preparing dataset...")
            progress.progress(10)

            X = df.drop(columns=[target])
            y = df[target]
            problem_type = infer_problem_type(y)

            if problem_type == "classification" and y.dtype == "object":
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                st.session_state.label_encoder = le
            else:
                st.session_state.label_encoder = None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42,
                stratify=y if problem_type == "classification" and len(np.unique(y)) > 1 else None
            )

            preprocessor = build_preprocessor(X)
            model, param_grid = get_model_and_grid(model_choice, problem_type)

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model),
            ])

            status.info("Running GridSearchCV...")
            progress.progress(35)
            scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=cv_folds,
                n_jobs=-1,
                scoring=scoring,
                verbose=0,
            )
            grid.fit(X_train, y_train)
            progress.progress(70)

            status.info("Running K-Fold cross-validation on best model...")
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_results = cross_validate(
                grid.best_estimator_,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy"] if problem_type == "classification" else ["neg_mean_squared_error", "r2"],
                n_jobs=-1,
            )
            progress.progress(85)

            status.info("Evaluating test performance...")
            metrics = evaluate_model(grid.best_estimator_, X_test, y_test, problem_type)
            progress.progress(100)
            status.success("Training complete.")

            st.session_state.trained_model = grid.best_estimator_
            st.session_state.model_name = model_choice
            st.session_state.model_type = problem_type
            st.session_state.target_column = target
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.feature_columns = X.columns.tolist()
            st.session_state.preprocessor = preprocessor
            st.session_state.metrics = metrics

            st.subheader("Best Parameters")
            st.json(grid.best_params_)

            st.subheader("Cross-Validation Results")
            if problem_type == "classification":
                st.write(f"CV Accuracy Mean: {cv_results['test_accuracy'].mean():.4f}")
            else:
                mse_mean = -cv_results["test_neg_mean_squared_error"].mean()
                st.write(f"CV MSE Mean: {mse_mean:.4f}")
                st.write(f"CV R² Mean: {cv_results['test_r2'].mean():.4f}")

            st.subheader("Test Metrics")
            st.json(metrics)

        except Exception as e:
            st.error(f"Training failed: {e}")


# --------------------------
# Page: Deep Learning
# --------------------------
def page_deep_learning() -> None:
    st.header("🤖 Deep Learning")

    if st.session_state.data is None:
        st.info("Please upload data first in Data Hub.")
        return

    df = st.session_state.data.copy()
    target = st.selectbox("Target Column", df.columns.tolist(), key="dl_target")

    st.subheader("Neural Network Builder")
    for i, layer in enumerate(st.session_state.dl_layers):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1:
            layer["units"] = st.number_input(f"Layer {i+1} Units", min_value=1, max_value=1024, value=int(layer["units"]), key=f"units_{i}")
        with c2:
            layer["activation"] = st.selectbox(f"Layer {i+1} Activation", ["relu", "tanh", "sigmoid", "linear"], index=["relu", "tanh", "sigmoid", "linear"].index(layer["activation"]), key=f"act_{i}")
        with c3:
            layer["dropout"] = st.slider(f"Layer {i+1} Dropout", 0.0, 0.8, float(layer["dropout"]), 0.05, key=f"drop_{i}")
        with c4:
            if st.button("❌", key=f"remove_{i}") and len(st.session_state.dl_layers) > 1:
                st.session_state.dl_layers.pop(i)
                st.rerun()

    if st.button("Add Layer"):
        st.session_state.dl_layers.append({"units": 32, "activation": "relu", "dropout": 0.0})
        st.rerun()

    epochs = st.slider("Epochs", 5, 200, 30)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    learning_rate = st.selectbox("Learning Rate", [1e-2, 1e-3, 1e-4], index=1)

    if st.button("Train Deep Learning Model"):
        try:
            X = df.drop(columns=[target])
            y = df[target]
            problem_type = infer_problem_type(y)

            X_num = pd.get_dummies(X, drop_first=True)
            X_num = X_num.fillna(X_num.median(numeric_only=True))

            if problem_type == "classification" and y.dtype == "object":
                le = LabelEncoder()
                y_enc = le.fit_transform(y.astype(str))
                st.session_state.label_encoder = le
            else:
                y_enc = y.values
                st.session_state.label_encoder = None

            X_train, X_test, y_train, y_test = train_test_split(
                X_num, y_enc, test_size=0.2, random_state=42,
                stratify=y_enc if problem_type == "classification" and len(np.unique(y_enc)) > 1 else None
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = Sequential()
            input_dim = X_train_scaled.shape[1]

            first = st.session_state.dl_layers[0]
            model.add(Dense(first["units"], activation=first["activation"], input_dim=input_dim))
            if first["dropout"] > 0:
                model.add(Dropout(first["dropout"]))

            for layer in st.session_state.dl_layers[1:]:
                model.add(Dense(layer["units"], activation=layer["activation"]))
                if layer["dropout"] > 0:
                    model.add(Dropout(layer["dropout"]))

            if problem_type == "classification":
                num_classes = len(np.unique(y_train))
                if num_classes <= 2:
                    model.add(Dense(1, activation="sigmoid"))
                    loss = "binary_crossentropy"
                    metrics = ["accuracy"]
                else:
                    model.add(Dense(num_classes, activation="softmax"))
                    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
                    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
                    loss = "categorical_crossentropy"
                    metrics = ["accuracy"]
            else:
                model.add(Dense(1, activation="linear"))
                loss = "mse"
                metrics = ["mae"]

            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            callbacks = [
                EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
            ]

            with st.spinner("Training neural network..."):
                history = model.fit(
                    X_train_scaled,
                    y_train,
                    validation_split=0.2,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0,
                )

            st.session_state.dl_model = {
                "keras_model": model,
                "scaler": scaler,
                "features": X_num.columns.tolist(),
                "problem_type": problem_type,
                "target": target,
            }
            st.session_state.dl_history = history.history

            st.success("Deep learning training complete.")

            hist_df = pd.DataFrame(history.history)
            fig = px.line(hist_df, y=[c for c in hist_df.columns if "loss" in c], title="Loss Curves")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Deep learning training failed: {e}")


# --------------------------
# Page: AI Assistant
# --------------------------
def page_ai_assistant() -> None:
    st.header("💬 AI Assistant")
    cfg = get_active_ai_config()

    providers = ["openai", "anthropic", "ollama", "custom"]
    default_provider = cfg.get("provider", "openai")
    provider = st.selectbox("Provider", providers, index=providers.index(default_provider) if default_provider in providers else 0)
    cfg["provider"] = provider

    status_text = "✅ Configured" if provider_is_configured(provider, cfg) else "⚠️ Missing required fields"
    st.caption(f"Current provider status ({provider}): {status_text}")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask anything...")
    if user_msg:
        if not provider_is_configured(provider, cfg):
            st.error(f"{provider} is not fully configured. Update it in Settings first.")
            return

        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            try:
                response = ai_request(provider, cfg, st.session_state.chat_history)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                err = f"Assistant request failed: {e}"
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err})

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------
# Page: Evaluation
# --------------------------
def page_evaluation() -> None:
    st.header("📈 Evaluation")
    if st.session_state.trained_model is None:
        st.info("Train a machine learning model first.")
        return

    model = st.session_state.trained_model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    problem_type = st.session_state.model_type

    if X_test is None or y_test is None:
        st.warning("No test data found in session.")
        return

    pred = model.predict(X_test)

    if problem_type == "classification":
        st.subheader("Classification Metrics")
        metrics = evaluate_model(model, X_test, y_test, problem_type)
        st.json(metrics)

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("ROC Curve")
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)
                if probs.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
                    roc_auc = auc(fpr, tpr)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}"))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
                    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("ROC curve visualization currently shown for binary classification only.")
            else:
                st.info("Model does not support probability predictions.")
        except Exception as e:
            st.warning(f"Could not compute ROC curve: {e}")

    else:
        st.subheader("Regression Metrics")
        st.json(evaluate_model(model, X_test, y_test, problem_type))

    st.subheader("Feature Importance")
    try:
        estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
            names = st.session_state.feature_columns or [f"f{i}" for i in range(len(importances))]
            fig = px.bar(x=names[:len(importances)], y=importances, labels={"x": "Feature", "y": "Importance"})
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(estimator, "coef_"):
            coef = np.ravel(estimator.coef_)
            names = st.session_state.feature_columns or [f"f{i}" for i in range(len(coef))]
            fig = px.bar(x=names[:len(coef)], y=coef, labels={"x": "Feature", "y": "Coefficient"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")
    except Exception as e:
        st.warning(f"Failed to display feature importance: {e}")


# --------------------------
# Page: Prediction
# --------------------------
def page_prediction() -> None:
    st.header("🔮 Prediction")

    if st.session_state.trained_model is None:
        st.info("Please train or load a model first.")
        return

    model = st.session_state.trained_model
    features = st.session_state.feature_columns or []

    st.subheader("Single Prediction")
    input_data = {}
    if st.session_state.data is not None and features:
        source = st.session_state.data
        for col in features:
            if col in source.columns:
                if pd.api.types.is_numeric_dtype(source[col]):
                    input_data[col] = st.number_input(f"{col}", value=float(source[col].median() if source[col].notna().any() else 0.0))
                else:
                    options = source[col].dropna().astype(str).unique().tolist()
                    input_data[col] = st.selectbox(f"{col}", options if options else [""])

        if st.button("Predict Single"):
            try:
                pred_df = pd.DataFrame([input_data])
                pred = model.predict(pred_df)[0]
                if st.session_state.label_encoder is not None:
                    pred = st.session_state.label_encoder.inverse_transform([int(pred)])[0]
                st.success(f"Prediction: {pred}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.subheader("Batch Prediction")
    batch_file = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="batch_pred")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            st.dataframe(batch_df.head(), use_container_width=True)
            if st.button("Run Batch Prediction"):
                preds = model.predict(batch_df)
                if st.session_state.label_encoder is not None:
                    preds = st.session_state.label_encoder.inverse_transform(preds.astype(int))
                out_df = batch_df.copy()
                out_df["prediction"] = preds
                st.dataframe(out_df.head(), use_container_width=True)
                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")


# --------------------------
# Page: Model Management
# --------------------------
def page_model_management() -> None:
    st.header("💾 Model Management")

    st.subheader("Save Current Model")
    if st.session_state.trained_model is None:
        st.info("No classical ML model available to save.")
    else:
        model_name = st.text_input("Model Filename", value=f"model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl")
        if st.button("Save Model"):
            try:
                model_path = os.path.join(MODEL_DIR, model_name)
                metadata_path = model_path + ".meta.json"

                payload = {
                    "model": st.session_state.trained_model,
                    "target_column": st.session_state.target_column,
                    "feature_columns": st.session_state.feature_columns,
                    "model_type": st.session_state.model_type,
                    "label_encoder": st.session_state.label_encoder,
                }

                with open(model_path, "wb") as f:
                    pickle.dump(payload, f)

                metadata = {
                    "id": str(uuid.uuid4()),
                    "file": model_name,
                    "saved_at_utc": datetime.utcnow().isoformat(),
                    "model_name": st.session_state.model_name,
                    "model_type": st.session_state.model_type,
                    "target_column": st.session_state.target_column,
                    "feature_count": len(st.session_state.feature_columns or []),
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                st.success(f"Model saved: {model_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    st.subheader("Load Model")
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")])
    if not model_files:
        st.info("No saved models found.")
    else:
        selected = st.selectbox("Select model file", model_files)
        if st.button("Load Selected Model"):
            try:
                path = os.path.join(MODEL_DIR, selected)
                with open(path, "rb") as f:
                    payload = pickle.load(f)

                st.session_state.trained_model = payload.get("model")
                st.session_state.target_column = payload.get("target_column")
                st.session_state.feature_columns = payload.get("feature_columns")
                st.session_state.model_type = payload.get("model_type")
                st.session_state.label_encoder = payload.get("label_encoder")

                st.success("Model loaded into session successfully.")
            except Exception as e:
                st.error(f"Load failed: {e}")

        st.subheader("Download Saved Models")
        for model_file in model_files:
            model_path = os.path.join(MODEL_DIR, model_file)
            metadata_path = model_path + ".meta.json"

            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f)
                except Exception:
                    metadata = {}

            with open(model_path, "rb") as f:
                model_bytes = f.read()

            download_name = build_model_download_name(model_file, metadata)
            metadata_filename = f"{os.path.splitext(download_name)[0]}.meta.json"
            metadata_json = json.dumps(metadata or {"file": model_file}, indent=2)

            st.markdown(f"**{model_file}**")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Model (.pkl)",
                    data=model_bytes,
                    file_name=download_name,
                    mime="application/octet-stream",
                    key=f"download_model_{model_file}",
                )
            with col2:
                st.download_button(
                    label="Download Metadata (.json)",
                    data=metadata_json,
                    file_name=metadata_filename,
                    mime="application/json",
                    key=f"download_meta_{model_file}",
                )

    st.subheader("Saved Model Metadata")
    meta_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".meta.json")]
    if meta_files:
        selected_meta = st.selectbox("Metadata File", meta_files)
        with open(os.path.join(MODEL_DIR, selected_meta), "r", encoding="utf-8") as f:
            st.json(json.load(f))


# --------------------------
# Page: Settings
# --------------------------
def page_settings() -> None:
    st.header("⚙️ Settings")
    cfg = get_active_ai_config()

    providers = ["openai", "anthropic", "ollama", "custom"]
    selected_provider = st.selectbox(
        "Default AI Provider",
        providers,
        index=providers.index(cfg.get("provider", "openai")) if cfg.get("provider", "openai") in providers else 0,
    )

    with st.expander("OpenAI", expanded=True):
        openai_api_key = st.text_input("OpenAI API Key", value=cfg["openai"].get("api_key", ""), type="password")
        openai_model = st.text_input("OpenAI Model Name", value=cfg["openai"].get("model", "gpt-4o-mini"))
        openai_base_url = st.text_input("OpenAI URL", value=cfg["openai"].get("base_url", "https://api.openai.com/v1/chat/completions"))

    with st.expander("Anthropic"):
        anthropic_api_key = st.text_input("Anthropic API Key", value=cfg["anthropic"].get("api_key", ""), type="password")
        anthropic_model = st.text_input("Anthropic Model Name", value=cfg["anthropic"].get("model", "claude-3-5-sonnet-20241022"))
        anthropic_base_url = st.text_input("Anthropic URL", value=cfg["anthropic"].get("base_url", "https://api.anthropic.com/v1/messages"))

    with st.expander("Ollama"):
        ollama_url = st.text_input("Ollama URL", value=cfg["ollama"].get("base_url", "http://localhost:11434/api/chat"))
        ollama_model = st.text_input("Ollama Model Name", value=cfg["ollama"].get("model", "llama3.1"))

    with st.expander("Custom API"):
        custom_url = st.text_input("Custom URL", value=cfg["custom"].get("base_url", ""))
        custom_api_key = st.text_input("Custom API Key", value=cfg["custom"].get("api_key", ""), type="password")
        custom_model = st.text_input("Custom Model Name", value=cfg["custom"].get("model", ""))
        custom_headers = st.text_area("Custom Headers (JSON)", value=cfg["custom"].get("headers", "{}"), height=80)
        custom_payload_template = st.text_area(
            "Payload Template (JSON, supports {prompt} and {model})",
            value=cfg["custom"].get("payload_template", '{"model":"{model}","messages":[{"role":"user","content":"{prompt}"}]}'),
            height=140,
        )

    if st.button("Save Configuration"):
        cfg["provider"] = selected_provider
        cfg["openai"]["api_key"] = openai_api_key
        cfg["openai"]["model"] = openai_model
        cfg["openai"]["base_url"] = openai_base_url

        cfg["anthropic"]["api_key"] = anthropic_api_key
        cfg["anthropic"]["model"] = anthropic_model
        cfg["anthropic"]["base_url"] = anthropic_base_url

        cfg["ollama"]["base_url"] = ollama_url
        cfg["ollama"]["model"] = ollama_model

        cfg["custom"]["base_url"] = custom_url
        cfg["custom"]["api_key"] = custom_api_key
        cfg["custom"]["model"] = custom_model
        cfg["custom"]["headers"] = custom_headers
        cfg["custom"]["payload_template"] = custom_payload_template

        st.session_state.ai_provider_config = cfg
        st.success("Configuration saved in session state.")

    st.subheader("Configuration Status")
    for p in providers:
        st.write(f"- **{p}**: {'✅ Configured' if provider_is_configured(p, cfg) else '❌ Not configured'}")


# --------------------------
# Page: User Guide
# --------------------------
def page_user_guide() -> None:
    st.header("📘 User Guide")
    st.markdown(
        """
### Welcome to AI LAB Advanced

#### Workflow
1. Go to **Data Hub** and upload your CSV/JSON dataset.
2. Use basic preprocessing and visualize the data.
3. Train a model in **ML Training** or build a neural net in **Deep Learning**.
4. Check performance in **Evaluation**.
5. Run single or batch inference in **Prediction**.
6. Save/load models from **Model Management**.
7. Configure AI providers in **Settings** and chat in **AI Assistant**.

#### Notes
- For classification tasks with text labels, labels are encoded automatically.
- Batch prediction expects columns aligned with training features.
- Deep learning currently trains on one-hot encoded tabular features.
- Use **Settings** to configure OpenAI, Anthropic, Ollama, or any custom API.

#### Troubleshooting
- If XGBoost fails, ensure `xgboost` is installed.
- If AI Assistant fails, validate API key, model, and URL.
- If prediction fails, confirm incoming columns match training features.
"""
    )


# --------------------------
# Main App
# --------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_session_state()

    st.title("🚀 AI LAB Advanced")
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        [
            "Data Hub",
            "ML Training",
            "Deep Learning",
            "AI Assistant",
            "Evaluation",
            "Prediction",
            "Model Management",
            "Settings",
            "User Guide",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Created by Mohammad Saeed Angiz**")

    if page == "Data Hub":
        page_data_hub()
    elif page == "ML Training":
        page_ml_training()
    elif page == "Deep Learning":
        page_deep_learning()
    elif page == "AI Assistant":
        page_ai_assistant()
    elif page == "Evaluation":
        page_evaluation()
    elif page == "Prediction":
        page_prediction()
    elif page == "Model Management":
        page_model_management()
    elif page == "Settings":
        page_settings()
    elif page == "User Guide":
        page_user_guide()

    st.markdown("---")
    st.caption("Created by Mohammad Saeed Angiz")


if __name__ == "__main__":
    main()
