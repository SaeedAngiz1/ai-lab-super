"""
AI Lab Super - Enterprise-Grade AI/ML/DL Development Platform
Complete Main Application with Python IDE and Jupyter Notebook IDE
Created by: Mohammad Saeed Angiz
All credits go to Mohammad Saeed Angiz
"""

import streamlit as st
import pandas as pd
import base64
from io import BytesIO
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="AI Lab Super",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
        # AI Lab Super
        **Created by: Mohammad Saeed Angiz**

        Enterprise-grade AI/ML/DL development platform.
        All credits go to Mohammad Saeed Angiz.

        Features:
        - Python IDE with AI Assistant
        - Jupyter Notebook IDE
        - Data Hub
        - ML Lab
        - DL Studio
        - Prediction Hub
        - And much more!
        """
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .creator-credit {
        color: #f0f0f0;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        width: 100%;
    }
    .download-button {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        text-decoration: none;
        display: inline-block;
    }
    .ai-assistant-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .quick-action-btn {
        margin-bottom: 0.5rem;
    }
    .provider-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)


# Helper function for downloads
def get_download_link(data, filename, file_format='csv'):
    """Generate download link for data"""
    href = ""
    if file_format == 'csv':
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">📥 Download CSV</a>'
    elif file_format == 'excel':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Sheet1')
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-button">📥 Download Excel</a>'
    elif file_format == 'json':
        json_str = data.to_json(orient='records')
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}" class="download-button">📥 Download JSON</a>'
    return href


# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data': None,
        'model': None,
        'project_name': "New Project",
        'experiments': [],
        'trained_models': {},
        'production_models': {},
        'ai_assistant_active': False,
        'hyperparameter_configs': {},
        'preprocessing_pipeline': None,
        'feature_engineering_config': {},
        'ai_log': [],
        'download_history': [],
        'current_process': {},
        'process_data_snapshots': {},
        'ai_commands_queue': [],
        'ide_files': {},
        'ide_current_file': None,
        'ide_output': [],
        'notebook_cells': [{
            'id': 0,
            'type': 'markdown',
            'content': '# Welcome to AI Lab Super Notebook\n\nCreated by Mohammad Saeed Angiz',
            'output': None,
            'execution_count': None
        }],
        'ai_provider': 'OpenAI',
        'ai_model': 'gpt-4',
        'custom_ai_model': '',
        'api_key': {},
        'api_endpoint': {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# Main header
st.markdown("""
<div class="main-header">
    <h1>🧪 AI Lab Super</h1>
    <div class="creator-credit">Created by: Mohammad Saeed Angiz | All Rights Reserved</div>
</div>
""", unsafe_allow_html=True)


# Helper function for AI commands (defined BEFORE it is used)
def execute_global_ai_command(command):
    """Execute AI command from any page"""
    if not command:
        st.sidebar.warning("⚠️ Please enter a command first.")
        return

    command_lower = command.lower()
    log_entry = f"[{datetime.now().strftime('%H:%M:%S')}] {command}"

    if "train" in command_lower or "model" in command_lower:
        st.session_state.ai_log.append(f"{log_entry} - Training initiated")
        st.session_state.ai_commands_queue.append(('train', command))
    elif "preprocess" in command_lower or "clean" in command_lower:
        st.session_state.ai_log.append(f"{log_entry} - Preprocessing initiated")
        st.session_state.ai_commands_queue.append(('preprocess', command))
    elif "download" in command_lower or "export" in command_lower:
        st.session_state.ai_log.append(f"{log_entry} - Download initiated")
        st.session_state.ai_commands_queue.append(('download', command))
    elif "deploy" in command_lower or "production" in command_lower:
        st.session_state.ai_log.append(f"{log_entry} - Deployment initiated")
        st.session_state.ai_commands_queue.append(('deploy', command))
    else:
        st.session_state.ai_log.append(f"{log_entry} - Command queued")
        st.session_state.ai_commands_queue.append(('custom', command))

    st.sidebar.success("✅ AI Command executed!")


# AI Provider Configuration
def show_ai_provider_config():
    """Show AI provider configuration in sidebar"""
    st.sidebar.markdown("## 🤖 AI Provider")
    st.sidebar.markdown("---")

    providers = ["OpenAI", "Anthropic", "Ollama", "Custom AI", "Azure OpenAI", "Google AI"]

    with st.sidebar.expander("⚙️ AI Configuration", expanded=True):
        # Safe index lookup
        current_provider = st.session_state.ai_provider
        provider_index = providers.index(current_provider) if current_provider in providers else 0

        provider = st.selectbox(
            "Select AI Provider",
            providers,
            key="ai_provider_select",
            index=provider_index
        )
        st.session_state.ai_provider = provider

        # Model selection based on provider
        model_options = {
            "OpenAI": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
            "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229",
                          "claude-3-haiku-20240307", "claude-2.1", "claude-2.0"],
            "Ollama": ["llama3", "llama2", "mistral", "codellama", "phi3", "gemma", "qwen2"],
            "Azure OpenAI": ["gpt-4", "gpt-35-turbo", "gpt-4o"],
            "Google AI": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-pro-vision"]
        }

        if provider == "Custom AI":
            # Free-text box for custom model name
            custom_model = st.text_input(
                "Enter Custom Model Name",
                value=st.session_state.get("custom_ai_model", ""),
                placeholder="e.g., my-finetuned-llm-v2",
                key="custom_ai_model_input"
            )
            st.session_state.custom_ai_model = custom_model
            st.session_state.ai_model = custom_model.strip() if custom_model else ""

            if not st.session_state.ai_model:
                st.caption("⚠️ Please type the name of your custom model.")
        else:
            options = model_options.get(provider, ["gpt-4"])
            st.session_state.ai_model = st.selectbox(
                "Select Model",
                options,
                key="ai_model_select"
            )

        # API Key input
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key.get(provider, ""),
            key=f"api_key_input_{provider}"
        )
        st.session_state.api_key[provider] = api_key

        # Custom endpoint for Custom AI and Ollama
        if provider in ["Custom AI", "Ollama"]:
            endpoint = st.text_input(
                "API Endpoint",
                value=st.session_state.api_endpoint.get(
                    provider,
                    "http://localhost:11434" if provider == "Ollama" else ""
                ),
                placeholder="https://api.your-provider.com/v1",
                key=f"endpoint_input_{provider}"
            )
            st.session_state.api_endpoint[provider] = endpoint

        # Test connection button
        if st.button("🔌 Test Connection", key="test_connection"):
            if provider == "Custom AI" and not st.session_state.ai_model:
                st.warning("⚠️ Please enter a custom model name.")
            elif provider == "Custom AI" and not st.session_state.api_endpoint.get(provider):
                st.warning("⚠️ Please enter the API endpoint.")
            elif api_key or provider == "Ollama":
                st.success(f"✅ {provider} configured successfully!")
                st.info(f"Model: {st.session_state.ai_model or 'N/A'}")
            else:
                st.warning("⚠️ Please enter an API key")

        # Show current configuration status
        st.markdown("---")
        st.markdown("**Current Config:**")
        st.markdown(f"Provider: `{provider}`")
        st.markdown(f"Model: `{st.session_state.ai_model or 'Not set'}`")
        if provider in ["Custom AI", "Ollama"]:
            st.markdown(f"Endpoint: `{st.session_state.api_endpoint.get(provider, 'Not set')}`")


# Global AI Assistant (Sidebar)
def show_global_ai_assistant():
    """Show global AI assistant in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🤖 AI Assistant")

    with st.sidebar.expander("🤖 AI Control Panel", expanded=False):
        st.markdown("### AI Commands")

        nl_command = st.text_area(
            "Tell AI what to do:",
            placeholder="e.g., 'Train best model and deploy to production'",
            key="global_ai_command",
            height=100
        )

        if st.button("🚀 Execute AI Command", key="global_execute"):
            execute_global_ai_command(nl_command)

        st.markdown("#### Quick Actions")
        st.markdown("---")

        if st.button("📊 Auto Process Data", key="auto_process", use_container_width=True):
            execute_global_ai_command("Auto process data with preprocessing")

        if st.button("🎯 Find Best Model", key="find_best", use_container_width=True):
            execute_global_ai_command("Find best model with cross-validation")

        if st.button("⚡ Quick Train", key="quick_train", use_container_width=True):
            execute_global_ai_command("Quick train with default settings")

        if st.button("🚀 Deploy Best Model", key="deploy_best", use_container_width=True):
            execute_global_ai_command("Deploy best model to production")

        if st.button("📊 Generate Report", key="gen_report", use_container_width=True):
            execute_global_ai_command("Generate comprehensive report")

        if st.button("💾 Save All Results", key="save_results", use_container_width=True):
            execute_global_ai_command("Save all results to files")

        st.markdown("---")
        st.markdown("#### AI Status")

        if st.session_state.ai_log:
            with st.expander("📝 Recent AI Actions", expanded=False):
                for log in st.session_state.ai_log[-5:]:
                    st.markdown(f"• {log}")

        st.markdown("#### AI Settings")
        st.checkbox("Auto-save Results", value=True, key="auto_save")
        st.checkbox("Auto-deploy Models", value=False, key="auto_deploy")
        st.checkbox("Notify on Complete", value=True, key="notify_complete")


# Download Center in Sidebar
def show_download_center():
    """Show download center in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📥 Download Center")

    if st.session_state.data is not None:
        st.sidebar.markdown("### Current Data")

        data_format = st.sidebar.selectbox(
            "Format",
            ["CSV", "Excel", "JSON"],
            key="download_format"
        )

        if st.sidebar.button("📥 Download Data", key="download_data_btn"):
            filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{data_format.lower()}"
            st.sidebar.markdown(
                get_download_link(st.session_state.data, filename, data_format.lower()),
                unsafe_allow_html=True
            )

    if st.session_state.trained_models:
        st.sidebar.markdown("### Trained Models")

        model_to_download = st.sidebar.selectbox(
            "Select Model",
            list(st.session_state.trained_models.keys()),
            key="download_model_select"
        )

        if st.sidebar.button("📥 Download Model Info", key="download_model_btn"):
            model_info = st.session_state.trained_models[model_to_download]
            model_df = pd.DataFrame([{
                'Model': model_to_download,
                'Task Type': model_info.get('task_type', 'N/A'),
                'Metrics': str(model_info.get('metrics', {})),
                'Training Time': model_info.get('training_time', 'N/A')
            }])
            st.sidebar.markdown(
                get_download_link(model_df, f"{model_to_download}_info.csv", 'csv'),
                unsafe_allow_html=True
            )

    if st.session_state.experiments:
        st.sidebar.markdown("### Experiments")

        if st.sidebar.button("📥 Download All Experiments", key="download_exp_btn"):
            exp_df = pd.DataFrame(st.session_state.experiments)
            st.sidebar.markdown(
                get_download_link(exp_df, "experiments.csv", 'csv'),
                unsafe_allow_html=True
            )

    if st.session_state.download_history:
        st.sidebar.markdown("### Download History")
        for entry in st.session_state.download_history[-3:]:
            st.sidebar.markdown(f"• {entry}")


# Navigation
st.sidebar.markdown("## 🎯 Navigation")
st.sidebar.markdown("---")

pages = {
    "🏠 Home": "home",
    "📊 Data Hub": "data_hub",
    "🤖 ML Lab": "ml_lab",
    "🧠 DL Studio": "dl_studio",
    "📈 Evaluation": "evaluation",
    "🔮 Prediction Hub": "prediction",
    "📁 Project Management": "project_mgmt",
    "💻 Python IDE": "python_ide",
    "📓 Jupyter Notebook": "jupyter_notebook",
    "🤖 AI Assistant": "ai_assistant",
    "📖 User Guide": "user_guide"
}

selection = st.sidebar.radio("Go to", list(pages.keys()))
current_page = pages[selection]

# Sidebar project info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Current Project")
st.session_state.project_name = st.sidebar.text_input(
    "Project Name", value=st.session_state.project_name, key="sidebar_project_name"
)

if st.sidebar.button("💾 Save Project"):
    st.session_state.project_saved = True
    st.sidebar.success("✅ Project saved!")

# Quick stats in sidebar
if st.session_state.data is not None:
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Rows", st.session_state.data.shape[0])
    st.sidebar.metric("Columns", st.session_state.data.shape[1])
    st.sidebar.metric("Missing Values", int(st.session_state.data.isnull().sum().sum()))

# Production models status
if st.session_state.production_models:
    st.sidebar.markdown("### 🚀 Production Models")
    for model_name, model_info in st.session_state.production_models.items():
        status = "🟢 Active" if model_info.get('active', False) else "🔴 Inactive"
        st.sidebar.markdown(f"**{model_name}**: {status}")

# Show global components
show_ai_provider_config()
show_download_center()
show_global_ai_assistant()

# Creator credit
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 8px; color: white;'>
    <small>Created by<br><b>Mohammad Saeed Angiz</b></small>
</div>
""", unsafe_allow_html=True)

# Page routing
if current_page == "home":
    st.markdown("## 🎉 Welcome to AI Lab Super!")
    st.markdown("**Created by: Mohammad Saeed Angiz**")

    st.markdown("### 📥 Quick Downloads")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.session_state.data is not None:
            st.markdown(get_download_link(st.session_state.data, "current_data.csv", 'csv'), unsafe_allow_html=True)
        else:
            st.info("No data loaded yet")

    with col2:
        if st.session_state.trained_models:
            model_names = list(st.session_state.trained_models.keys())
            model_summary = pd.DataFrame({'Model': model_names})
            st.markdown(get_download_link(model_summary, "model_summary.csv", 'csv'), unsafe_allow_html=True)
        else:
            st.info("No models trained yet")

    with col3:
        if st.session_state.experiments:
            exp_df = pd.DataFrame(st.session_state.experiments)
            st.markdown(get_download_link(exp_df, "experiments.csv", 'csv'), unsafe_allow_html=True)
        else:
            st.info("No experiments yet")

    st.markdown("---")
    st.markdown("## 🚀 Features Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📊 Data Hub
        - Upload & manage datasets
        - Generate synthetic data
        - Data preprocessing
        - Feature engineering
        - Data visualization
        - **Download at any stage**
        """)

    with col2:
        st.markdown("""
        ### 🤖 ML Lab
        - Multiple ML algorithms
        - AutoML support
        - Hyperparameter tuning
        - Cross-validation
        - Ensemble methods
        - **Download models & results**
        """)

    with col3:
        st.markdown("""
        ### 🧠 DL Studio
        - Neural network design
        - Transfer learning
        - Model training
        - Real-time monitoring
        - Model export
        - **Download trained models**
        """)

    st.markdown("---")
    st.markdown("## 💻 System Status")

    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        st.metric("🧪 Experiments", len(st.session_state.experiments))
    with status_col2:
        st.metric("🤖 Trained Models", len(st.session_state.trained_models))
    with status_col3:
        st.metric("📊 Datasets Loaded", 1 if st.session_state.data is not None else 0)
    with status_col4:
        st.metric("🚀 Production Models", len(st.session_state.production_models))

elif current_page == "data_hub":
    try:
        st.switch_page("pages/01_Data_Hub.py")
    except Exception:
        st.markdown("## 📊 Data Hub")
        st.markdown("**Created by: Mohammad Saeed Angiz**")
        st.info("Data Hub page is available at pages/01_Data_Hub.py")

elif current_page == "ml_lab":
    try:
        st.switch_page("pages/02_ML_Lab.py")
    except Exception:
        st.markdown("## 🤖 ML Lab")
        st.markdown("**Created by: Mohammad Saeed Angiz**")
        st.info("ML Lab page is available at pages/02_ML_Lab.py")

elif current_page == "python_ide":
    try:
        st.switch_page("pages/11_Python_IDE.py")
    except Exception:
        st.markdown("## 💻 Python IDE")
        st.markdown("**Created by: Mohammad Saeed Angiz**")
        st.info("Python IDE page is available at pages/11_Python_IDE.py")

elif current_page == "jupyter_notebook":
    try:
        st.switch_page("pages/12_Jupyter_IDE.py")
    except Exception:
        st.markdown("## 📓 Jupyter Notebook IDE")
        st.markdown("**Created by: Mohammad Saeed Angiz**")
        st.info("Jupyter Notebook IDE page is available at pages/12_Jupyter_IDE.py")

elif current_page == "dl_studio":
    try:
        st.switch_page("pages/06_DL_LAB.py")
    except Exception:
        st.markdown("## 🧠 Deep Learning Studio")
        st.markdown("**Created by: Mohammad Saeed Angiz**")
        st.info("DL Studio page is available at pages/06_DL_LAB.py")

elif current_page == "evaluation":
    st.markdown("## 📈 Model Evaluation")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    st.info("Comprehensive model evaluation and metrics analysis.")
    if st.session_state.trained_models:
        st.markdown("### Trained Models Available")
        for model_name in st.session_state.trained_models.keys():
            st.markdown(f"- {model_name}")
    else:
        st.warning("No trained models available for evaluation.")

elif current_page == "prediction":
    st.markdown("## 🔮 Prediction Hub")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    st.info("Make predictions using your trained models.")
    if st.session_state.production_models:
        st.markdown("### Active Models")
        for model_name, info in st.session_state.production_models.items():
            st.markdown(f"- {model_name}: {'🟢 Active' if info.get('active') else '🔴 Inactive'}")
    else:
        st.warning("No models deployed to production yet.")

elif current_page == "project_mgmt":
    st.markdown("## 📁 Project Management")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    st.info("Manage your AI/ML projects efficiently.")

    st.markdown("### Current Project")
    new_name = st.text_input("Project Name", value=st.session_state.project_name, key="pm_project_name")
    st.text_area("Project Description", key="project_desc")

    if st.button("💾 Save Project Settings"):
        st.session_state.project_name = new_name
        st.success("✅ Project settings saved!")

elif current_page == "ai_assistant":
    st.markdown("## 🤖 AI Assistant")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    st.info("AI-powered assistant for your ML/DL workflow.")

    st.markdown(f"**Current AI Provider:** {st.session_state.ai_provider}")
    st.markdown(f"**Current Model:** {st.session_state.ai_model or 'Not set'}")

    user_input = st.text_area("Ask the AI Assistant:", height=150)
    if st.button("Send to AI"):
        if user_input:
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**AI ({st.session_state.ai_model or 'unconfigured'}):** "
                        f"AI response will be processed using {st.session_state.ai_provider}...")
            st.info("Configure your API key in the sidebar AI Provider section to enable full AI functionality.")

elif current_page == "user_guide":
    st.markdown("## 📖 User Guide")
    st.markdown("**Created by: Mohammad Saeed Angiz**")

    st.markdown("""
    ### Getting Started

    1. **Configure AI Provider**: Set up your AI provider in the sidebar
    2. **Load Data**: Navigate to Data Hub to upload your dataset
    3. **Train Models**: Use ML Lab to train machine learning models
    4. **Evaluate**: Check model performance in Evaluation
    5. **Deploy**: Deploy models to production for predictions

    ### AI Providers Supported
    - **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4o
    - **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
    - **Ollama**: Local models (Llama 3, Mistral, etc.)
    - **Custom AI**: Type **any** model name + your own API endpoint
    - **Azure OpenAI**: Azure-hosted OpenAI models
    - **Google AI**: Gemini models

    ### Tips
    - Use the AI Assistant for natural language commands
    - Quick Actions automate common tasks
    - Download your results at any stage
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background: #4b0082; border-radius: 8px;'>
    <p><b>AI Lab Super</b> | Created by <b>Mohammad Saeed Angiz</b></p>
    <p>All credits and rights belong to <b>Mohammad Saeed Angiz</b></p>
    <p>Version 1.1.0 | © 2026</p>
    <p><b>Including:</b> Python IDE • Jupyter Notebook IDE • AI Assistant • Complete ML/DL Pipeline • Multi-Provider AI Support</p>
</div>
""", unsafe_allow_html=True)