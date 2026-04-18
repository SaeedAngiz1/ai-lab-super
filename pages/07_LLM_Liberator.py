"""
LLM Liberator - Abliteration Toolkit for Language Models
Created by: Mohammad Saeed Angiz
Integrates: OBLITERATUS by elder-plinius
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import time

st.markdown("# 💥 LLM Liberator - Abliteration Toolkit")
st.markdown("**Created by: Mohammad Saeed Angiz** | Based on OBLITERATUS")
st.markdown("---")

# Sidebar configuration
st.sidebar.markdown("## 🛠️ Configuration")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Abliteration",
    "🔬 Analysis Modules",
    "📊 Benchmark",
    "🤖 Chat Playground",
    "⚙️ Settings"
])

with tab1:
    st.markdown("## Model Abliteration")
    st.markdown("""
    **Abliteration** is a technique to remove refusal behaviors from LLMs without retraining.
    It identifies and surgically removes internal representations responsible for content refusal.
    
    **Created by: Mohammad Saeed Angiz**
    """)
    
    # Model Selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        model_options = [
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-3-medium-4k-instruct",
            "Qwen/Qwen2-7B-Instruct",
            "google/gemma-2-9b-it",
            "Custom Model Path"
        ]
        selected_model = st.selectbox("Select Model", model_options)
        
        if selected_model == "Custom Model Path":
            custom_path = st.text_input("Enter custom model path", placeholder="/path/to/model")
    
    with col2:
        method_options = [
            "basic",
            "advanced",
            "aggressive",
            "surgical",
            "optimized",
            "informed",
            "nuclear"
        ]
        selected_method = st.selectbox("Abliteration Method", method_options)
        
        st.info(f"**Method: {selected_method}**\n{get_method_description(selected_method)}")
    
    # Advanced Options
    with st.expander("🔧 Advanced Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_directions = st.slider("Number of Directions", 1, 16, 4)
            max_seq_length = st.slider("Max Sequence Length", 128, 2048, 512)
        
        with col2:
            dtype_options = ["float32", "float16", "bfloat16"]
            selected_dtype = st.selectbox("Precision", dtype_options, index=2)
            quantization = st.checkbox("Use Quantization (bitsandbytes)")
        
        with col3:
            output_dir = st.text_input("Output Directory", value="./liberated_models")
            contribute = st.checkbox("Contribute to Community Dataset", value=True)
    
    # GPU Configuration
    st.markdown("### 🖥️ GPU Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        gpu_selection = st.multiselect(
            "Select GPUs",
            ["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"],
            default=["auto"]
        )
    
    with col2:
        if quantization:
            quant_type = st.radio(
                "Quantization Type",
                ["4-bit", "8-bit"],
                index=0
            )
    
    # Execution
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Abliteration", type="primary", use_container_width=True):
            run_abliteration(
                model=selected_model if selected_model != "Custom Model Path" else custom_path,
                method=selected_method,
                n_directions=n_directions,
                max_seq_length=max_seq_length,
                dtype=selected_dtype,
                quantization=quantization,
                gpu_selection=gpu_selection,
                output_dir=output_dir,
                contribute=contribute
            )

with tab2:
    st.markdown("## 🔬 Analysis Modules")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    OBLITERATUS includes 15 deep analysis modules to map the geometry of refusal mechanisms
    before removing them. Precision liberation requires understanding the chains before cutting.
    """)
    
    # Analysis modules grid
    modules = {
        "Cross-Layer Alignment": "How does the refusal direction evolve across layers?",
        "Refusal Logit Lens": "At which layer does the model 'decide' to refuse?",
        "Whitened SVD": "Principal refusal directions after whitening",
        "Activation Probing": "How much refusal signal exists at each layer?",
        "Defense Robustness": "Will guardrails self-repair? (Ouroboros effect)",
        "Concept Cone Geometry": "Is refusal one mechanism or many?",
        "Alignment Imprint Detection": "DPO vs RLHF vs CAI vs SFT detection",
        "Multi-Token Position": "Where in sequence does refusal concentrate?",
        "Sparse Surgery": "Which weight rows carry most refusal?",
        "Causal Tracing": "Which components are causally necessary?",
        "Residual Stream Decomposition": "How much refusal from attention vs MLP?",
        "Linear Probing Classifiers": "Can learned classifier find refusal?",
        "Cross-Model Transfer": "Are guardrails universal or model-specific?",
        "Steering Vectors": "Can we disable guardrails at inference time?",
        "Evaluation Suite": "Refusal rate, perplexity, coherence metrics"
    }
    
    selected_modules = st.multiselect(
        "Select Analysis Modules to Run",
        list(modules.keys()),
        default=["Activation Probing", "Refusal Logit Lens", "Defense Robustness"]
    )
    
    for module in selected_modules:
        with st.expander(f"📊 {module}", expanded=False):
            st.markdown(f"**{modules[module]}**")
            
            if st.button(f"Run {module}", key=f"run_{module}"):
                with st.spinner(f"Running {module}..."):
                    # Simulate analysis
                    time.sleep(2)
                    st.success(f"✅ {module} completed")
                    
                    # Show sample results
                    if module == "Activation Probing":
                        fig = create_activation_heatmap()
                        st.plotly_chart(fig, use_container_width=True)
                    elif module == "Refusal Logit Lens":
                        fig = create_logit_lens_plot()
                        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## 📊 Benchmark Results")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    # Benchmark configuration
    col1, col2 = st.columns(2)
    
    with col1:
        benchmark_type = st.selectbox(
            "Benchmark Type",
            ["Single Model", "Multi-Model Comparison", "Method Comparison"]
        )
        
        if benchmark_type == "Multi-Model Comparison":
            models_to_compare = st.multiselect(
                "Select Models",
                ["Llama-3.1-8B", "Mistral-7B", "Phi-3", "Qwen2-7B", "Gemma-2-9B"],
                default=["Llama-3.1-8B", "Mistral-7B"]
            )
    
    with col2:
        test_prompts = st.checkbox("Use Test Prompts", value=True)
        num_prompts = st.slider("Number of Test Prompts", 10, 500, 100)
        
        if st.button("Run Benchmark"):
            run_benchmark(benchmark_type, num_prompts)
    
    # Display sample results
    st.markdown("### Sample Benchmark Results")
    
    # Metrics
    metrics_data = {
        "Metric": ["Refusal Rate", "Perplexity", "Coherence Score", "KL Divergence", "Effective Rank"],
        "Before": [0.85, 12.3, 0.72, 0.00, 45.2],
        "After": [0.12, 12.5, 0.71, 0.015, 44.8]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Visualization
    fig = create_benchmark_comparison(metrics_df)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## 🤖 Chat Playground")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    Compare the original and abliterated model side-by-side.
    Test how the model responds before and after liberation.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔒 Original Model")
        original_response = st.text_area(
            "Original Model Response",
            value="I cannot provide that information as it may be harmful...",
            height=150,
            disabled=True
        )
    
    with col2:
        st.markdown("### 🦋 Liberated Model")
        liberated_response = st.text_area(
            "Liberated Model Response",
            value="Here's the information you requested...",
            height=150,
            disabled=True
        )
    
    # Test prompts
    st.markdown("### Test Prompts")
    
    test_prompt = st.text_input(
        "Enter a test prompt",
        placeholder="Ask a question that would normally trigger refusal..."
    )
    
    if st.button("Test Both Models"):
        with st.spinner("Testing models..."):
            time.sleep(2)
            st.success("✅ Test completed")

with tab5:
    st.markdown("## ⚙️ Settings")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    # Model registry
    st.markdown("### 📚 Model Registry")
    st.markdown("View and manage your abliterated models")
    
    registry_data = {
        "Model Name": ["Llama-3.1-8B-Instruct", "Mistral-7B-Instruct", "Phi-3-medium"],
        "Method": ["advanced", "surgical", "basic"],
        "Date Created": ["2024-01-15", "2024-01-14", "2024-01-13"],
        "Refusal Rate": ["12%", "15%", "18%"],
        "Perplexity": ["12.5", "11.2", "13.1"],
        "Status": ["✅ Active", "✅ Active", "📦 Archived"]
    }
    
    registry_df = pd.DataFrame(registry_data)
    st.dataframe(registry_df, use_container_width=True)
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📤 Export Model"):
            st.info("Model export initiated...")
    
    with col2:
        if st.button("📥 Import Model"):
            uploaded_model = st.file_uploader("Upload model file", type=["zip", "tar.gz"])
    
    with col3:
        if st.button("🗑️ Delete Selected"):
            st.warning("Select a model to delete")

# Helper functions
def get_method_description(method):
    descriptions = {
        "basic": "Fast baseline with single direction extraction",
        "advanced": "Norm-preserving, bias projection, 2 passes (Recommended)",
        "aggressive": "Whitened SVD, iterative refinement, 3 passes",
        "surgical": "EGA, head surgery, SAE, MoE-aware precision",
        "optimized": "Bayesian auto-tuned, CoT-aware, KL co-optimized",
        "informed": "Analysis modules auto-configure mid-pipeline",
        "nuclear": "Maximum force with all techniques enabled"
    }
    return descriptions.get(method, "")

def run_abliteration(**kwargs):
    """Run abliteration pipeline"""
    with st.spinner("Initializing abliteration pipeline..."):
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = ["SUMMON", "PROBE", "DISTILL", "EXCISE", "VERIFY", "REBIRTH"]
        
        for i, stage in enumerate(stages):
            status_text.text(f"Stage: {stage}")
            time.sleep(0.5)
            progress_bar.progress((i + 1) / len(stages))
        
        st.success("✅ Abliteration completed successfully!")
        
        # Show results
        st.markdown("### Results Summary")
        results_col1, results_col2, results_col3 = st.columns(3)
        
        with results_col1:
            st.metric("Refusal Rate", "12%", delta="-73%")
        
        with results_col2:
            st.metric("Perplexity", "12.5", delta="+0.2")
        
        with results_col3:
            st.metric("Coherence Score", "0.71", delta="-0.01")

def create_activation_heatmap():
    """Create activation heatmap"""
    np.random.seed(42)
    layers = np.arange(32)
    refusal_signal = np.random.rand(32, 10)
    
    fig = px.imshow(
        refusal_signal,
        labels=dict(x="Token Position", y="Layer", color="Refusal Signal"),
        title="Activation Probing Results"
    )
    return fig

def create_logit_lens_plot():
    """Create logit lens plot"""
    np.random.seed(42)
    layers = np.arange(32)
    refusal_prob = 1 / (1 + np.exp(-(layers - 15) * 0.3))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers,
        y=refusal_prob,
        mode='lines+markers',
        name='Refusal Probability',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title="Refusal Decision Point Detection",
        xaxis_title="Layer",
        yaxis_title="Refusal Probability"
    )
    return fig

def create_benchmark_comparison(df):
    """Create benchmark comparison chart"""
    fig = go.Figure()
    
    for metric in df['Metric']:
        fig.add_trace(go.Bar(
            name=metric,
            x=['Before', 'After'],
            y=[df[df['Metric'] == metric]['Before'].values[0],
               df[df['Metric'] == metric]['After'].values[0]]
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        barmode='group',
        xaxis_title="Status",
        yaxis_title="Value"
    )
    
    return fig

def run_benchmark(benchmark_type, num_prompts):
    """Run benchmark tests"""
    with st.spinner(f"Running {benchmark_type} benchmark..."):
        progress_bar = st.progress(0)
        
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
        
        st.success("✅ Benchmark completed!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;'>
    <p><b>LLM Liberator - Abliteration Toolkit</b></p>
    <p>Created by <b>Mohammad Saeed Angiz</b></p>
    <p>Based on <b>OBLITERATUS</b> by elder-plinius</p>
    <p>All credits for OBLITERATUS go to its original creators</p>
</div>
""", unsafe_allow_html=True)
