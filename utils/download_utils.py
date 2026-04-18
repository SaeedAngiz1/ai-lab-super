"""
Download Utilities - Global Download Functions
Created by: Mohammad Saeed Angiz
"""

import streamlit as st
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import json
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

class DownloadManager:
    """Global download manager for all pages"""
    
    @staticmethod
    def download_dataframe(df, filename_prefix="data", formats=["csv", "excel", "json"]):
        """Download dataframe in multiple formats"""
        st.markdown("### 📥 Download Options")
        
        cols = st.columns(len(formats))
        
        for i, fmt in enumerate(formats):
            with cols[i]:
                filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{fmt}"
                
                if fmt == "csv":
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button" style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #28a745 0%, #20c997 100%); color: white; border-radius: 5px; text-decoration: none;">📥 Download CSV</a>'
                
                elif fmt == "excel":
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                    b64 = base64.b64encode(output.getvalue()).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-button" style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #007bff 0%, #0056b3 100%); color: white; border-radius: 5px; text-decoration: none;">📥 Download Excel</a>'
                
                elif fmt == "json":
                    json_str = df.to_json(orient='records', indent=2)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="{filename}" class="download-button" style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #17a2b8 0%, #138496 100%); color: white; border-radius: 5px; text-decoration: none;">📥 Download JSON</a>'
                
                st.markdown(href, unsafe_allow_html=True)
    
    @staticmethod
    def download_model_info(model_info, model_name):
        """Download model information"""
        st.markdown("### 📥 Download Model Info")
        
        # Create model summary
        summary = {
            'Model Name': model_name,
            'Task Type': model_info.get('task_type', 'N/A'),
            'Training Time': model_info.get('training_time', 'N/A'),
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add metrics
        if 'metrics' in model_info:
            for key, value in model_info['metrics'].items():
                summary[f'Metric_{key}'] = value
        
        # Add hyperparameters
        if 'best_params' in model_info:
            for key, value in model_info['best_params'].items():
                summary[f'Param_{key}'] = value
        
        summary_df = pd.DataFrame([summary])
        
        filename = f"model_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv = summary_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 5px; text-decoration: none;">📥 Download Model Info</a>'
        
        st.markdown(href, unsafe_allow_html=True)
    
    @staticmethod
    def download_plotly_figure(fig, filename_prefix="plot"):
        """Download Plotly figure as HTML or PNG"""
        st.markdown("### 📥 Download Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as HTML
            html_string = fig.to_html()
            b64 = base64.b64encode(html_string.encode()).decode()
            filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            href = f'<a href="data:text/html;base64,{b64}" download="{filename}" style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #28a745 0%, #20c997 100%); color: white; border-radius: 5px; text-decoration: none;">📥 Download HTML</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Download as PNG (static)
            img_bytes = fig.to_image(format="png")
            b64 = base64.b64encode(img_bytes).decode()
            filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            href = f'<a href="data:image/png;base64,{b64}" download="{filename}" style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #007bff 0%, #0056b3 100%); color: white; border-radius: 5px; text-decoration: none;">📥 Download PNG</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    @staticmethod
    def download_experiment_results(results, experiment_name):
        """Download experiment results"""
        st.markdown("### 📥 Download Experiment Results")
        
        # Convert results to DataFrame
        if isinstance(results, list):
            results_df = pd.DataFrame(results)
        elif isinstance(results, dict):
            results_df = pd.DataFrame([results])
        else:
            results_df = results
        
        filename = f"experiment_{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv = results_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display: inline-block; padding: 10px 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 5px; text-decoration: none;">📥 Download Results</a>'
        
        st.markdown(href, unsafe_allow_html=True)

class DataPreviewManager:
    """Manager for data preview at any processing stage"""
    
    @staticmethod
    def show_preview_with_download(data, stage_name, show_download=True):
        """Show data preview with download option"""
        st.markdown(f"### 📊 Data Preview - {stage_name}")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Missing Values", data.isnull().sum().sum())
        col4.metric("Memory (MB)", f"{data.memory_usage(deep=True).sum() / 1024**2:.2f}")
        
        # Data preview
        preview_rows = st.slider(f"Rows to display ({stage_name})", 5, 50, 20, key=f"preview_{stage_name}")
        st.dataframe(data.head(preview_rows), use_container_width=True)
        
        # Download option
        if show_download:
            DownloadManager.download_dataframe(
                data, 
                filename_prefix=stage_name.replace(" ", "_").lower(),
                formats=["csv", "excel", "json"]
            )
    
    @staticmethod
    def show_processing_comparison(data_before, data_after, process_name):
        """Show before/after comparison of processing"""
        st.markdown(f"### 🔄 Processing Comparison - {process_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Before")
            st.metric("Rows", data_before.shape[0])
            st.metric("Columns", data_before.shape[1])
            st.metric("Missing Values", data_before.isnull().sum().sum())
        
        with col2:
            st.markdown("#### After")
            st.metric("Rows", data_after.shape[0])
            st.metric("Columns", data_after.shape[1])
            st.metric("Missing Values", data_after.isnull().sum().sum())
        
        # Download both
        st.markdown("### 📥 Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            DownloadManager.download_dataframe(
                data_before,
                filename_prefix=f"{process_name}_before",
                formats=["csv"]
            )
        
        with col2:
            DownloadManager.download_dataframe(
                data_after,
                filename_prefix=f"{process_name}_after",
                formats=["csv"]
            )

# Integration function
def add_download_section_to_page(data=None, model_info=None, results=None, page_name=""):
    """Add download section to any page"""
    st.markdown("---")
    st.markdown("## 📥 Download Center")
    
    if data is not None:
        st.markdown("### Data")
        DataPreviewManager.show_preview_with_download(data, page_name)
    
    if model_info is not None:
        st.markdown("### Model")
        DownloadManager.download_model_info(model_info, page_name)
    
    if results is not None:
        st.markdown("### Results")
        DownloadManager.download_experiment_results(results, page_name)
