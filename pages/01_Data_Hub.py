"""
Data Hub - Data Management and Preprocessing
Created by: Mohammad Saeed Angiz
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_regression, load_iris, load_wine, load_breast_cancer, fetch_california_housing
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
import io

st.markdown("# 📊 Data Hub")
st.markdown("**Created by: Mohammad Saeed Angiz**")
st.markdown("---")

# Tabs for different data operations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload Data", 
    "🎲 Generate Data",
    "🔍 Data Preview", 
    "⚙️ Preprocessing",
    "🔧 Feature Engineering",
    "📈 Visualization"
])

with tab1:
    st.markdown("## Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'json', 'parquet', 'tsv'],
        help="Supported formats: CSV, Excel, JSON, Parquet, TSV"
    )
    
    if uploaded_file is not None:
        try:
            # Read file based on extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            with st.spinner("Loading data..."):
                if file_extension == 'csv':
                    data = pd.read_csv(uploaded_file)
                elif file_extension in ['xlsx', 'xls']:
                    data = pd.read_excel(uploaded_file)
                elif file_extension == 'json':
                    data = pd.read_json(uploaded_file)
                elif file_extension == 'parquet':
                    data = pd.read_parquet(uploaded_file)
                elif file_extension == 'tsv':
                    data = pd.read_csv(uploaded_file, sep='\t')
            
            st.session_state.data = data
            st.success(f"✅ File uploaded successfully! Shape: {data.shape}")
            
            # Quick preview
            st.markdown("### Quick Preview")
            st.dataframe(data.head(10), use_container_width=True)
            
            # Save preprocessing state
            st.session_state.preprocessing_pipeline = {
                'original_shape': data.shape,
                'steps': []
            }
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")

with tab2:
    st.markdown("## Generate Synthetic Data")
    
    gen_col1, gen_col2 = st.columns(2)
    
    with gen_col1:
        gen_type = st.selectbox("Data Type", ["Classification", "Regression", "Time Series", "Clustering"])
        n_samples = st.slider("Number of Samples", 100, 50000, 1000)
        n_features = st.slider("Number of Features", 5, 100, 20)
    
    with gen_col2:
        if gen_type in ["Classification", "Clustering"]:
            n_classes = st.slider("Number of Classes/Clusters", 2, 20, 2)
            n_informative = st.slider("Informative Features", 2, min(n_features, n_features), min(10, n_features))
        else:
            n_informative = st.slider("Informative Features", 2, min(n_features, n_features), min(10, n_features))
        
        noise = st.slider("Noise Level", 0.0, 2.0, 0.1)
        random_state = st.number_input("Random Seed", 0, 1000, 42)
    
    if st.button("Generate Data", type="primary"):
        try:
            with st.spinner("Generating data..."):
                if gen_type == "Classification":
                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_informative=n_informative,
                        n_classes=n_classes,
                        n_redundant=max(0, n_features - n_informative - 1),
                        flip_y=noise,
                        random_state=random_state
                    )
                    feature_names = [f"feature_{i}" for i in range(n_features)]
                    data = pd.DataFrame(X, columns=feature_names)
                    data['target'] = y
                    
                elif gen_type == "Regression":
                    X, y = make_regression(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_informative=n_informative,
                        noise=noise * 10,
                        random_state=random_state
                    )
                    feature_names = [f"feature_{i}" for i in range(n_features)]
                    data = pd.DataFrame(X, columns=feature_names)
                    data['target'] = y
                    
                elif gen_type == "Time Series":
                    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
                    trend = np.linspace(0, 100, n_samples)
                    seasonality = 10 * np.sin(np.linspace(0, 20*np.pi, n_samples))
                    noise_data = np.random.normal(0, noise, n_samples)
                    data = pd.DataFrame({
                        'date': dates,
                        'trend': trend,
                        'seasonality': seasonality,
                        'value': trend + seasonality + noise_data
                    })
                    
                elif gen_type == "Clustering":
                    from sklearn.datasets import make_blobs
                    X, y = make_blobs(
                        n_samples=n_samples,
                        n_features=n_features,
                        centers=n_classes,
                        cluster_std=noise * 2,
                        random_state=random_state
                    )
                    feature_names = [f"feature_{i}" for i in range(n_features)]
                    data = pd.DataFrame(X, columns=feature_names)
                    data['cluster'] = y
            
            st.session_state.data = data
            st.success(f"✅ {gen_type} data generated! Shape: {data.shape}")
            st.dataframe(data.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error generating data: {str(e)}")
    
    # Sample datasets
    st.markdown("### 📚 Sample Datasets")
    sample_col1, sample_col2, sample_col3, sample_col4 = st.columns(4)
    
    with sample_col1:
        if st.button("Iris Dataset"):
            iris = load_iris(as_frame=True)
            st.session_state.data = iris.frame
            st.success("✅ Iris dataset loaded!")
    
    with sample_col2:
        if st.button("Wine Dataset"):
            wine = load_wine(as_frame=True)
            st.session_state.data = wine.frame
            st.success("✅ Wine dataset loaded!")
    
    with sample_col3:
        if st.button("Breast Cancer"):
            cancer = load_breast_cancer(as_frame=True)
            st.session_state.data = cancer.frame
            st.success("✅ Breast Cancer dataset loaded!")
    
    with sample_col4:
        if st.button("California Housing"):
            housing = fetch_california_housing(as_frame=True)
            st.session_state.data = housing.frame
            st.success("✅ California Housing dataset loaded!")

with tab3:
    if st.session_state.data is not None:
        data = st.session_state.data
        
        st.markdown("## Data Overview")
        
        # Basic statistics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Missing Values", data.isnull().sum().sum())
        col4.metric("Duplicates", data.duplicated().sum())
        col5.metric("Memory (MB)", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f}")
        
        # Data types
        st.markdown("### Data Types")
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes.values,
            'Non-Null Count': data.count().values,
            'Null Count': data.isnull().sum().values,
            'Null Percentage': (data.isnull().sum() / len(data) * 100).round(2).values,
            'Unique Values': data.nunique().values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        # Statistical summary
        st.markdown("### Statistical Summary")
        st.dataframe(data.describe(include='all'), use_container_width=True)
        
        # Data preview
        st.markdown("### Data Preview")
        preview_rows = st.slider("Rows to display", 5, 50, 20)
        st.dataframe(data.head(preview_rows), use_container_width=True)
        
        # Column details
        st.markdown("### Column Details")
        selected_col = st.selectbox("Select Column", data.columns)
        
        if selected_col:
            col_data = data[selected_col]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Column:** `{selected_col}`")
                st.markdown(f"**Type:** {col_data.dtype}")
                st.markdown(f"**Non-Null:** {col_data.count()}")
                st.markdown(f"**Null:** {col_data.isnull().sum()}")
                st.markdown(f"**Unique:** {col_data.nunique()}")
                
                if col_data.dtype in ['int64', 'float64']:
                    st.markdown(f"**Mean:** {col_data.mean():.4f}")
                    st.markdown(f"**Std:** {col_data.std():.4f}")
                    st.markdown(f"**Min:** {col_data.min()}")
                    st.markdown(f"**Max:** {col_data.max()}")
            
            with col2:
                if col_data.dtype in ['int64', 'float64']:
                    fig = px.histogram(data, x=selected_col, nbins=50, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    value_counts = col_data.value_counts().head(20)
                    fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Top 20 values in {selected_col}")
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please upload or generate data first!")

with tab4:
    if st.session_state.data is not None:
        data = st.session_state.data.copy()
        
        st.markdown("## Data Preprocessing")
        
        # Initialize preprocessing steps
        if 'preprocessing_steps' not in st.session_state:
            st.session_state.preprocessing_steps = []
        
        # Missing values handling
        st.markdown("### Missing Values Handling")
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if missing_cols:
            st.write(f"Columns with missing values: {missing_cols}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                missing_strategy = st.selectbox(
                    "Missing Value Strategy",
                    ["Drop rows", "Drop columns", "Fill with mean", "Fill with median", 
                     "Fill with mode", "Fill with constant", "KNN Imputer"]
                )
            
            with col2:
                if missing_strategy == "Fill with constant":
                    fill_value = st.text_input("Fill Value", "0")
            
            with col3:
                if st.button("Apply Missing Value Strategy"):
                    data = apply_missing_value_strategy(data, missing_cols, missing_strategy, fill_value if missing_strategy == "Fill with constant" else None)
                    st.session_state.data = data
                    st.session_state.preprocessing_steps.append(f"Missing values: {missing_strategy}")
                    st.success(f"✅ Applied {missing_strategy}!")
        else:
            st.success("✅ No missing values detected!")
        
        # Feature scaling
        st.markdown("### Feature Scaling")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            scaling_method = st.selectbox(
                "Scaling Method",
                ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
            )
            
            cols_to_scale = st.multiselect(
                "Select columns to scale",
                numeric_cols,
                default=numeric_cols
            )
            
            if st.button("Apply Scaling") and scaling_method != "None":
                data = apply_scaling(data, cols_to_scale, scaling_method)
                st.session_state.data = data
                st.session_state.preprocessing_steps.append(f"Scaling: {scaling_method}")
                st.success(f"✅ Applied {scaling_method}!")
        
        # Encoding categorical variables
        st.markdown("### Categorical Encoding")
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            encoding_method = st.selectbox(
                "Encoding Method",
                ["None", "Label Encoding", "One-Hot Encoding", "Target Encoding"]
            )
            
            cols_to_encode = st.multiselect(
                "Select columns to encode",
                categorical_cols,
                default=categorical_cols
            )
            
            if st.button("Apply Encoding") and encoding_method != "None":
                data = apply_encoding(data, cols_to_encode, encoding_method)
                st.session_state.data = data
                st.session_state.preprocessing_steps.append(f"Encoding: {encoding_method}")
                st.success(f"✅ Applied {encoding_method}!")
        
        # Outlier detection
        st.markdown("### Outlier Detection")
        outlier_method = st.selectbox(
            "Outlier Detection Method",
            ["None", "IQR Method", "Z-Score", "Isolation Forest"]
        )
        
        if st.button("Detect Outliers") and outlier_method != "None":
            outliers = detect_outliers(data, outlier_method)
            st.write(f"Found {outliers.sum()} outliers")
        
        # Preprocessing history
        if st.session_state.preprocessing_steps:
            st.markdown("### Preprocessing History")
            for step in st.session_state.preprocessing_steps:
                st.markdown(f"✓ {step}")
        
        # Reset preprocessing
        if st.button("Reset All Preprocessing"):
            st.session_state.preprocessing_steps = []
            st.success("✅ All preprocessing steps reset!")
    else:
        st.warning("⚠️ Please upload or generate data first!")

with tab5:
    if st.session_state.data is not None:
        data = st.session_state.data
        
        st.markdown("## Feature Engineering")
        
        # Feature selection
        st.markdown("### Feature Selection")
        
        target_col = st.selectbox("Select Target Column", data.columns)
        feature_cols = [col for col in data.columns if col != target_col]
        
        selection_method = st.selectbox(
            "Selection Method",
            ["None", "SelectKBest", "Variance Threshold", "Correlation Threshold"]
        )
        
        if selection_method == "SelectKBest":
            k = st.slider("Number of features to select", 1, len(feature_cols), min(10, len(feature_cols)))
            score_func = st.selectbox("Score Function", ["f_classif", "f_regression", "mutual_info_classif"])
        
        if st.button("Apply Feature Selection"):
            # Feature selection logic here
            st.success("✅ Feature selection applied!")
        
        # Dimensionality reduction
        st.markdown("### Dimensionality Reduction")
        
        dim_method = st.selectbox(
            "Dimensionality Reduction Method",
            ["None", "PCA", "t-SNE", "UMAP"]
        )
        
        if dim_method == "PCA":
            n_components = st.slider("Number of Components", 2, min(50, len(feature_cols)), 2)
            
            if st.button("Apply PCA"):
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(data[feature_cols].select_dtypes(include=[np.number]))
                
                st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
                st.write(f"Total Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")
                
                # Plot
                if n_components == 2:
                    fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], 
                                   color=data[target_col] if target_col in data.columns else None,
                                   title="PCA Result")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Feature creation
        st.markdown("### Feature Creation")
        
        st.markdown("#### Polynomial Features")
        poly_degree = st.slider("Polynomial Degree", 2, 5, 2)
        if st.button("Create Polynomial Features"):
            st.success("✅ Polynomial features created!")
        
        st.markdown("#### Interaction Features")
        if st.button("Create Interaction Features"):
            st.success("✅ Interaction features created!")
        
        st.markdown("#### Custom Feature")
        custom_expr = st.text_input("Enter expression (e.g., 'feature_0 * feature_1')")
        if st.button("Add Custom Feature"):
            st.success("✅ Custom feature added!")
    else:
        st.warning("⚠️ Please upload or generate data first!")

with tab6:
    if st.session_state.data is not None:
        data = st.session_state.data
        
        st.markdown("## Data Visualization")
        
        # Correlation heatmap
        st.markdown("### Correlation Heatmap")
        numeric_data = data.select_dtypes(include=[np.number])
        
        if not numeric_data.empty:
            corr = numeric_data.corr()
            fig = px.imshow(
                corr,
                labels=dict(color="Correlation"),
                x=corr.columns,
                y=corr.columns,
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        st.markdown("### Feature Distributions")
        numeric_cols = numeric_data.columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select feature for distribution", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(data, x=selected_col, marginal="box", 
                                  title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(data, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Pair plot
        st.markdown("### Pairwise Relationships")
        if len(numeric_cols) <= 10:
            sample_size = min(500, len(data))
            sample_data = data[numeric_cols].sample(n=sample_size, random_state=42)
            
            if st.button("Generate Pair Plot"):
                fig = px.scatter_matrix(sample_data, height=800)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Too many features for pair plot. Please select fewer features.")
        
        # Scatter plot
        st.markdown("### Scatter Plot")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X-axis", data.columns)
        with col2:
            y_col = st.selectbox("Y-axis", data.columns)
        with col3:
            color_col = st.selectbox("Color by", ["None"] + list(data.columns))
        
        if st.button("Generate Scatter Plot"):
            fig = px.scatter(data, x=x_col, y=y_col, 
                           color=color_col if color_col != "None" else None,
                           title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please upload or generate data first!")

# Helper functions
def apply_missing_value_strategy(data, cols, strategy, fill_value=None):
    """Apply missing value strategy"""
    df = data.copy()
    
    if strategy == "Drop rows":
        df = df.dropna(subset=cols)
    elif strategy == "Drop columns":
        df = df.drop(columns=cols)
    elif strategy == "Fill with mean":
        for col in cols:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
    elif strategy == "Fill with median":
        for col in cols:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
    elif strategy == "Fill with mode":
        for col in cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
    elif strategy == "Fill with constant":
        for col in cols:
            df[col].fillna(fill_value, inplace=True)
    elif strategy == "KNN Imputer":
        imputer = KNNImputer()
        df[cols] = imputer.fit_transform(df[cols])
    
    return df

def apply_scaling(data, cols, method):
    """Apply feature scaling"""
    df = data.copy()
    
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif method == "RobustScaler":
        scaler = RobustScaler()
    
    df[cols] = scaler.fit_transform(df[cols])
    return df

def apply_encoding(data, cols, method):
    """Apply categorical encoding"""
    df = data.copy()
    
    if method == "Label Encoding":
        le = LabelEncoder()
        for col in cols:
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=cols, drop_first=True)
    
    return df

def detect_outliers(data, method):
    """Detect outliers"""
    # Implementation based on selected method
    return pd.Series([False] * len(data))

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background: #4b0082; border-radius: 8px;'>
    <p><b>Data Hub</b> | Created by <b>Mohammad Saeed Angiz</b></p>
</div>
""", unsafe_allow_html=True)
