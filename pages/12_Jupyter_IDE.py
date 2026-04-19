import streamlit as st
import os
import json
from datetime import datetime
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import traceback
import warnings
import time
import re
import base64
from io import BytesIO
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Configure matplotlib backend before importing pyplot
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    plt.ioff()  # Turn off interactive mode
    # Suppress the non-interactive backend warning
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg is non-interactive.*')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# App metadata
APP_VERSION = "v2.0"
APP_CREATOR = "Mohammad Saeed Angiz"
APP_NAME = "SuperNote AI"

# Page configuration
# Try to use the icon file, fallback to emoji
icon_path = None
if os.path.exists("SuperNote AI.png"):
    icon_path = "SuperNote AI.png"

st.set_page_config(
    page_title=APP_NAME,
    page_icon=icon_path if icon_path else "🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'workspace_path' not in st.session_state:
    st.session_state.workspace_path = os.path.join(os.getcwd(), "workspace")
    os.makedirs(st.session_state.workspace_path, exist_ok=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'saved_files' not in st.session_state:
    st.session_state.saved_files = []

if 'notebook_cells' not in st.session_state:
    st.session_state.notebook_cells = [{
        "id": 0, 
        "code": "", 
        "output": "", 
        "error": "", 
        "executed": False,
        "cell_type": "code",  # code, markdown, raw
        "execution_count": 0,
        "execution_time": None,
        "collapsed": False,
        "metadata": {}
    }]
else:
    # Migrate old cells to new format
    for cell in st.session_state.notebook_cells:
        if "cell_type" not in cell:
            cell["cell_type"] = "code"
        if "execution_count" not in cell:
            cell["execution_count"] = 0
        if "execution_time" not in cell:
            cell["execution_time"] = None
        if "collapsed" not in cell:
            cell["collapsed"] = False
        if "metadata" not in cell:
            cell["metadata"] = {}

if 'notebook_variables' not in st.session_state:
    st.session_state.notebook_variables = {}

if 'notebook_metadata' not in st.session_state:
    st.session_state.notebook_metadata = {
        "name": "Untitled",
        "created": datetime.now().isoformat(),
        "modified": datetime.now().isoformat(),
        "creator": APP_CREATOR,
        "version": APP_VERSION
    }

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434/api"
DEFAULT_MODEL = "llama3.2"  # Change to your preferred model

def call_ollama(prompt, model=DEFAULT_MODEL, system_prompt=None, use_chat=False):
    """Call Ollama API for AI responses"""
    try:
        if use_chat:
            url = f"{OLLAMA_BASE_URL}/chat"
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
        else:
            url = f"{OLLAMA_BASE_URL}/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            if system_prompt:
                payload["system"] = system_prompt
        
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
            if use_chat:
                return result.get("message", {}).get("content", "")
            else:
                return result.get("response", "")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running on localhost:11434"
    except Exception as e:
        return f"Error: {str(e)}"

def get_code_suggestions(code_context):
    """Get AI code suggestions based on context"""
    prompt = f"""Based on this code context, provide helpful code suggestions or completions:
    
{code_context}

Provide only the code suggestion without explanations."""
    
    system_prompt = "You are a helpful coding assistant. Provide concise, accurate code suggestions."
    return call_ollama(prompt, system_prompt=system_prompt)

def autocorrect_code(code):
    """Autocorrect code using AI"""
    prompt = f"""Review and correct any errors in this code. Return the corrected code:

{code}

Return only the corrected code without explanations."""
    
    system_prompt = "You are a code reviewer. Fix syntax errors, logical issues, and improve code quality."
    return call_ollama(prompt, system_prompt=system_prompt)

def generate_code(description):
    """Generate code from description"""
    prompt = f"""Generate complete, working code based on this description:

{description}

Provide the complete code with comments."""
    
    system_prompt = "You are an expert programmer. Generate clean, well-documented, production-ready code."
    return call_ollama(prompt, system_prompt=system_prompt)

def generate_visualization(data_description, data_type="sample"):
    """Generate visualization code"""
    prompt = f"""Generate Python code using plotly or matplotlib to create a visualization based on:
- Data description: {data_description}
- Data type: {data_type}

Provide complete, runnable code that creates the visualization."""
    
    system_prompt = "You are a data visualization expert. Generate code that creates beautiful, informative visualizations."
    return call_ollama(prompt, system_prompt=system_prompt)

def generate_app(app_description):
    """Generate a complete Streamlit app"""
    prompt = f"""Generate a complete Streamlit application based on this description:

{app_description}

Include all necessary imports, UI components, and functionality. Make it production-ready."""
    
    system_prompt = "You are an expert Streamlit developer. Generate complete, functional Streamlit applications."
    return call_ollama(prompt, system_prompt=system_prompt)

def generate_presentation(data_description, data=None, presentation_type="business", auto_visualize=True, selected_viz=None):
    """Generate a business presentation with AI-selected visualizations"""
    if data is not None and isinstance(data, pd.DataFrame):
        data_summary = f"""
Data Summary:
- Shape: {data.shape}
- Columns: {', '.join(data.columns.tolist())}
- Sample data:
{data.head().to_string()}
"""
    else:
        data_summary = f"Data description: {data_description}"
    
    viz_selection = ""
    if auto_visualize:
        viz_selection = "Automatically select the best visualizations based on the data."
    elif selected_viz:
        viz_selection = f"Use these visualizations: {', '.join(selected_viz)}"
    
    prompt = f"""Create a professional {presentation_type} presentation about this data:

{data_summary}

Requirements:
- Create a comprehensive presentation with multiple slides
- {viz_selection}
- Include key insights and recommendations
- Make it suitable for {presentation_type} audience
- Use appropriate visualizations (bar charts, line charts, pie charts, scatter plots, etc.)
- Include data-driven insights
- Add actionable recommendations

Generate Python code that:
1. Creates visualizations using plotly
2. Organizes them into a presentation format
3. Includes markdown text for each slide
4. Saves the presentation as HTML

Return complete, runnable code."""
    
    system_prompt = "You are an expert data analyst and presentation designer. Create professional, data-driven presentations with optimal visualizations."
    return call_ollama(prompt, system_prompt=system_prompt)

def download_plotly_figure(fig, format="png"):
    """Convert plotly figure to downloadable format"""
    if format == "png":
        return fig.to_image(format="png", width=1200, height=800)
    elif format == "html":
        return fig.to_html()
    elif format == "pdf":
        # For PDF, we'll need to use kaleido or convert from PNG
        try:
            return fig.to_image(format="png", width=1200, height=800)
        except:
            return None
    return None

def process_magic_commands(code):
    """Process Jupyter magic commands"""
    lines = code.split('\n')
    processed_lines = []
    magic_results = {}
    
    for line in lines:
        # Line magic commands
        if line.strip().startswith('%'):
            magic = line.strip()
            if magic.startswith('%time'):
                # Time execution
                magic_results['time'] = True
                processed_lines.append(line.replace('%time', '# Timing execution'))
            elif magic.startswith('%timeit'):
                # Timeit execution
                magic_results['timeit'] = True
                processed_lines.append(line.replace('%timeit', '# Timing execution (timeit)'))
            elif magic.startswith('%matplotlib'):
                # Matplotlib backend
                if 'inline' in magic:
                    processed_lines.append("import matplotlib; matplotlib.use('Agg')")
                else:
                    processed_lines.append(line)
            elif magic.startswith('%load'):
                # Load from file
                filepath = magic.split()[1] if len(magic.split()) > 1 else ''
                if filepath and os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        processed_lines.append(f.read())
                else:
                    processed_lines.append(f"# Could not load {filepath}")
            elif magic.startswith('%who'):
                # List variables
                magic_results['who'] = True
                processed_lines.append("print('Variables:', list(globals().keys()))")
            elif magic.startswith('%whos'):
                # Detailed variable info
                magic_results['whos'] = True
                processed_lines.append("print('Variables:', {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_')})")
            else:
                processed_lines.append(line)
        # Cell magic commands
        elif line.strip().startswith('%%'):
            magic = line.strip()
            if magic.startswith('%%time'):
                magic_results['time'] = True
                processed_lines.append('# Timing cell execution')
            elif magic.startswith('%%timeit'):
                magic_results['timeit'] = True
                processed_lines.append('# Timing cell execution (timeit)')
            elif magic.startswith('%%writefile'):
                # Write to file
                filepath = magic.split()[1] if len(magic.split()) > 1 else 'output.txt'
                magic_results['writefile'] = filepath
                processed_lines.append(f"# Writing to {filepath}")
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines), magic_results

def execute_code(code, cell_id):
    """Execute Python code and capture output with timing and magic commands"""
    if not code.strip():
        return "", "", None
    
    start_time = time.time()
    
    # Process magic commands
    processed_code, magic_results = process_magic_commands(code)
    
    # Suppress matplotlib warnings
    warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg is non-interactive.*')
    warnings.filterwarnings('ignore', category=UserWarning, message='.*Matplotlib is currently using agg.*')
    
    # Create a namespace for this execution
    namespace = {
        'pd': pd,
        'px': px,
        'go': go,
        '__builtins__': __builtins__,
        'display': lambda *args, **kwargs: print(*args) if args else None,  # Simple display function
        **st.session_state.notebook_variables
    }
    
    # Try to import common libraries
    try:
        namespace['np'] = __import__('numpy')
    except ImportError:
        pass
    
    try:
        if MATPLOTLIB_AVAILABLE and plt:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as cell_plt
            cell_plt.ioff()
            def show_override(*args, **kwargs):
                pass
            cell_plt.show = show_override
            namespace['plt'] = cell_plt
    except:
        pass
    
    try:
        namespace['sns'] = __import__('seaborn')
    except ImportError:
        pass
    
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    error = ""
    output = ""
    execution_time = None
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture), warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*FigureCanvasAgg is non-interactive.*')
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Matplotlib is currently using agg.*')
            warnings.simplefilter('ignore', UserWarning)
            exec(processed_code, namespace)
        
        execution_time = time.time() - start_time
        
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        
        # Handle magic command results
        if magic_results.get('writefile'):
            # Write output to file
            filepath = magic_results['writefile']
            with open(os.path.join(st.session_state.workspace_path, filepath), 'w') as f:
                f.write(output)
            output += f"\n# Written to {filepath}"
        
        if magic_results.get('time') or magic_results.get('timeit'):
            output += f"\n# Execution time: {execution_time:.4f} seconds"
        
        # Filter out matplotlib warnings from stderr
        if error_output:
            lines = error_output.split('\n')
            filtered_lines = [
                line for line in lines 
                if 'FigureCanvasAgg is non-interactive' not in line
                and 'Matplotlib is currently using agg' not in line
            ]
            error_output = '\n'.join(filtered_lines)
            if error_output.strip():
                error = error_output
        
        # Update global variables (except special ones)
        excluded = ['pd', 'px', 'go', 'np', 'plt', 'sns', 'st', 'display']
        for key, value in namespace.items():
            if not key.startswith('__') and key not in excluded:
                st.session_state.notebook_variables[key] = value
                
    except Exception as e:
        execution_time = time.time() - start_time
        error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    
    return output, error, execution_time

# Sidebar navigation
st.sidebar.title("🚀 AI Workspace")
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Code Editor", "AI Chat", "Code Generator", "Visualization Generator", 
     "Presentation Generator", "App Generator", "File Manager", "Workspace"]
)

# Home Page
if page == "Home":
    # Header with icon
    col_header1, col_header2 = st.columns([1, 4])
    with col_header1:
        if icon_path and os.path.exists(icon_path):
            st.image(icon_path, width=64)
    with col_header2:
        st.title(f"📓 {APP_NAME}")
        st.markdown(f"**Version:** {APP_VERSION} | **Created by:** {APP_CREATOR}")
    
    st.markdown("### Integrated with Ollama AI")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Workspace", "Active", "✓")
    with col2:
        st.metric("AI Model", DEFAULT_MODEL, "Ready")
    with col3:
        st.metric("Files Saved", len(st.session_state.saved_files), "📁")
    
    st.markdown("---")
    
    # Creator info box
    with st.container():
        st.info(f"👨‍💻 **Created by:** {APP_CREATOR} | **Version:** {APP_VERSION}")
    
    st.markdown("""
    ### Features:
    - ✨ **AI Suggestions & Autocomplete**: Get intelligent code suggestions
    - 🔧 **Autocorrection**: Fix code errors automatically
    - 💻 **Code Generation**: Generate code from descriptions
    - 📊 **Visualization Generator**: Create charts and graphs with download options
    - 📈 **AI Presentation Generator**: Create professional business presentations with AI-selected visualizations
    - 🎨 **App Generator**: Build complete Streamlit apps
    - 💾 **File Management**: Save data, create folders
    - 📂 **Workspace**: Organize your projects
    """)
    
    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📝 New Code File", use_container_width=True):
            st.session_state.current_page = "Code Editor"
            st.rerun()
    with col2:
        if st.button("💬 Start AI Chat", use_container_width=True):
            st.session_state.current_page = "AI Chat"
            st.rerun()
    with col3:
        if st.button("📊 Create Visualization", use_container_width=True):
            st.session_state.current_page = "Visualization Generator"
            st.rerun()

# Code Editor Page (Full Jupyter Notebook Implementation)
elif page == "Code Editor":
    st.title("📓 Jupyter Notebook Editor")
    
    # Notebook metadata
    with st.expander("📋 Notebook Info", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.notebook_metadata["name"] = st.text_input(
                "Notebook Name", 
                value=st.session_state.notebook_metadata.get("name", "Untitled")
            )
        with col2:
            st.text(f"Created: {st.session_state.notebook_metadata.get('created', 'N/A')}")
            st.text(f"Modified: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.text(f"App Version: {APP_VERSION}")
            st.text(f"Creator: {APP_CREATOR}")
    
    # Enhanced Toolbar
    toolbar_cols = st.columns([1, 1, 1, 1, 1, 1, 1, 2])
    
    with toolbar_cols[0]:
        cell_type = st.selectbox("Cell Type", ["Code", "Markdown", "Raw"], key="new_cell_type")
        if st.button("➕ Add", use_container_width=True):
            new_id = max([cell["id"] for cell in st.session_state.notebook_cells], default=-1) + 1
            cell_type_map = {"Code": "code", "Markdown": "markdown", "Raw": "raw"}
            st.session_state.notebook_cells.append({
                "id": new_id,
                "code": "",
                "output": "",
                "error": "",
                "executed": False,
                "cell_type": cell_type_map[cell_type],
                "execution_count": 0,
                "execution_time": None,
                "collapsed": False,
                "metadata": {}
            })
            st.rerun()
    
    with toolbar_cols[1]:
        if st.button("▶️ Run All", use_container_width=True):
            execution_count = 1
            for cell in st.session_state.notebook_cells:
                if cell.get("cell_type", "code") == "code" and cell.get("code", "").strip():
                    output, error, exec_time = execute_code(cell["code"], cell["id"])
                    cell["output"] = output
                    cell["error"] = error
                    cell["executed"] = True
                    cell["execution_count"] = execution_count
                    cell["execution_time"] = exec_time
                    execution_count += 1
            st.rerun()
    
    with toolbar_cols[2]:
        if st.button("🔄 Clear Outputs", use_container_width=True):
            for cell in st.session_state.notebook_cells:
                cell["output"] = ""
                cell["error"] = ""
                cell["executed"] = False
                cell["execution_count"] = 0
                cell["execution_time"] = None
            st.rerun()
    
    with toolbar_cols[3]:
        if st.button("⬆️⬇️ Reorder", use_container_width=True):
            st.session_state.show_reorder = not st.session_state.get("show_reorder", False)
    
    with toolbar_cols[4]:
        export_format = st.selectbox("Export", [".py", ".ipynb", ".html"], key="export_format")
    
    with toolbar_cols[5]:
        if st.button("💾 Export", use_container_width=True):
            if export_format == ".py":
                all_code = "\n\n# Cell " + "\n\n# Cell ".join([
                    f"{i+1}\n{cell['code']}" 
                    for i, cell in enumerate(st.session_state.notebook_cells) 
                    if cell.get("code", "").strip() and cell.get("cell_type", "code") == "code"
                ])
                filename = st.session_state.notebook_metadata["name"] + ".py"
                filepath = os.path.join(st.session_state.workspace_path, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(all_code)
                st.success(f"Exported to {filepath}")
            elif export_format == ".ipynb":
                # Create Jupyter notebook format
                notebook = {
                    "cells": [],
                    "metadata": {
                        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python", "version": "3.8.0"}
                    },
                    "nbformat": 4,
                    "nbformat_minor": 4
                }
                for cell in st.session_state.notebook_cells:
                    cell_type = cell.get("cell_type", "code")
                    if cell_type == "code":
                        notebook["cells"].append({
                            "cell_type": "code",
                            "execution_count": cell.get("execution_count", None),
                            "metadata": cell.get("metadata", {}),
                            "outputs": [
                                {
                                    "output_type": "stream",
                                    "name": "stdout",
                                    "text": cell.get("output", "").split('\n')
                                } if cell.get("output") else {
                                    "output_type": "error",
                                    "ename": "Error",
                                    "evalue": cell.get("error", ""),
                                    "traceback": cell.get("error", "").split('\n')
                                } if cell.get("error") else {}
                            ],
                            "source": cell.get("code", "").split('\n')
                        })
                    elif cell_type == "markdown":
                        notebook["cells"].append({
                            "cell_type": "markdown",
                            "metadata": cell.get("metadata", {}),
                            "source": cell.get("code", "").split('\n')
                        })
                filename = st.session_state.notebook_metadata["name"] + ".ipynb"
                filepath = os.path.join(st.session_state.workspace_path, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(notebook, f, indent=2)
                st.success(f"Exported to {filepath}")
    
    with toolbar_cols[6]:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.notebook_cells = [{
                "id": 0, 
                "code": "", 
                "output": "", 
                "error": "", 
                "executed": False,
                "cell_type": "code",
                "execution_count": 0,
                "execution_time": None,
                "collapsed": False,
                "metadata": {}
            }]
            st.session_state.notebook_variables = {}
            st.rerun()
    
    st.markdown("---")
    
    # Display cells with full Jupyter features
    cells_to_remove = []
    cells_to_move = {}
    
    for idx, cell in enumerate(st.session_state.notebook_cells):
        # Cell container with border
        cell_expanded = not cell.get("collapsed", False)
        
        with st.container():
            # Cell header with execution count
            header_cols = st.columns([1, 6, 1, 1, 1, 1, 1])
            
            with header_cols[0]:
                # Execution count badge (like Jupyter [1], [2])
                if cell.get("executed") and cell.get("cell_type", "code") == "code":
                    exec_count = cell.get("execution_count", idx + 1)
                    st.markdown(f"**[{exec_count}]**")
                else:
                    st.markdown("**[ ]**")
            
            with header_cols[1]:
                cell_type_display = cell.get("cell_type", "code").title()
                st.markdown(f"**{cell_type_display} Cell {idx + 1}**")
                exec_time = cell.get("execution_time")
                if exec_time is not None:
                    st.caption(f"⏱️ {exec_time:.4f}s")
            
            with header_cols[2]:
                # Cell type selector
                current_type = cell.get("cell_type", "code")
                try:
                    type_index = ["code", "markdown", "raw"].index(current_type)
                except ValueError:
                    type_index = 0
                    current_type = "code"
                    cell["cell_type"] = "code"
                
                new_type = st.selectbox(
                    "", 
                    ["code", "markdown", "raw"],
                    index=type_index,
                    key=f"type_{cell['id']}",
                    label_visibility="collapsed"
                )
                if new_type != current_type:
                    cell["cell_type"] = new_type
                    st.rerun()
            
            with header_cols[3]:
                if st.button("▶️", key=f"run_{cell['id']}", use_container_width=True, help="Run cell"):
                    if cell.get("cell_type", "code") == "code" and cell.get("code", "").strip():
                        output, error, exec_time = execute_code(cell["code"], cell["id"])
                        cell["output"] = output
                        cell["error"] = error
                        cell["executed"] = True
                        cell["execution_count"] = max([c.get("execution_count", 0) for c in st.session_state.notebook_cells], default=0) + 1
                        cell["execution_time"] = exec_time
                        st.rerun()
            
            with header_cols[4]:
                if idx > 0:
                    if st.button("⬆️", key=f"up_{cell['id']}", use_container_width=True, help="Move up"):
                        st.session_state.notebook_cells[idx], st.session_state.notebook_cells[idx-1] = \
                            st.session_state.notebook_cells[idx-1], st.session_state.notebook_cells[idx]
                        st.rerun()
            
            with header_cols[5]:
                if idx < len(st.session_state.notebook_cells) - 1:
                    if st.button("⬇️", key=f"down_{cell['id']}", use_container_width=True, help="Move down"):
                        st.session_state.notebook_cells[idx], st.session_state.notebook_cells[idx+1] = \
                            st.session_state.notebook_cells[idx+1], st.session_state.notebook_cells[idx]
                        st.rerun()
            
            with header_cols[6]:
                if len(st.session_state.notebook_cells) > 1:
                    if st.button("🗑️", key=f"delete_{cell['id']}", use_container_width=True, help="Delete"):
                        cells_to_remove.append(cell["id"])
                if st.button("📂", key=f"collapse_{cell['id']}", use_container_width=True, help="Collapse"):
                    cell["collapsed"] = not cell.get("collapsed", False)
                    st.rerun()
            
            # Cell content based on type
            if cell.get("cell_type", "code") == "markdown":
                # Markdown cell - render as markdown
                if cell_expanded:
                    cell["code"] = st.text_area(
                        "",
                        value=cell["code"],
                        height=100,
                        key=f"code_{cell['id']}",
                        placeholder="Enter Markdown here...",
                        label_visibility="collapsed"
                    )
                    if cell["code"].strip():
                        st.markdown(cell["code"])
            else:
                # Code or Raw cell
                if cell_expanded:
                    cell["code"] = st.text_area(
                        "",
                        value=cell["code"],
                        height=200,
                        key=f"code_{cell['id']}",
                        placeholder="Enter Python code here...",
                        label_visibility="collapsed"
                    )
            
            # AI assistance toolbar (only for code cells)
            if cell.get("cell_type", "code") == "code" and cell_expanded:
                ai_cols = st.columns([1, 1, 1, 1, 1])
                with ai_cols[0]:
                    if st.button("✨ Suggest", key=f"suggest_{cell['id']}", use_container_width=True):
                        if cell["code"]:
                            with st.spinner("Getting suggestions..."):
                                suggestions = get_code_suggestions(cell["code"])
                                st.session_state[f"cell_suggestion_{cell['id']}"] = suggestions
                
                with ai_cols[1]:
                    if st.button("🔧 Fix", key=f"correct_{cell['id']}", use_container_width=True):
                        if cell["code"]:
                            with st.spinner("Correcting..."):
                                corrected = autocorrect_code(cell["code"])
                                st.session_state[f"cell_corrected_{cell['id']}"] = corrected
                
                with ai_cols[2]:
                    if st.button("💡 Complete", key=f"complete_{cell['id']}", use_container_width=True):
                        if cell["code"]:
                            with st.spinner("Completing..."):
                                completion = get_code_suggestions(cell["code"] + "\n# Continue:")
                                st.session_state[f"cell_completion_{cell['id']}"] = completion
                
                with ai_cols[3]:
                    if st.button("📝 Split", key=f"split_{cell['id']}", use_container_width=True):
                        # Split cell at cursor (middle for now)
                        lines = cell["code"].split('\n')
                        mid = len(lines) // 2
                        new_code1 = '\n'.join(lines[:mid])
                        new_code2 = '\n'.join(lines[mid:])
                        cell["code"] = new_code1
                        new_id = max([c["id"] for c in st.session_state.notebook_cells], default=-1) + 1
                        st.session_state.notebook_cells.insert(idx + 1, {
                            "id": new_id,
                            "code": new_code2,
                            "output": "",
                            "error": "",
                            "executed": False,
                            "cell_type": "code",
                            "execution_count": 0,
                            "execution_time": None,
                            "collapsed": False,
                            "metadata": {}
                        })
                        st.rerun()
                
                with ai_cols[4]:
                    if idx > 0 and st.session_state.notebook_cells[idx-1].get("cell_type", "code") == "code":
                        if st.button("🔗 Merge", key=f"merge_{cell['id']}", use_container_width=True):
                            prev_cell = st.session_state.notebook_cells[idx-1]
                            cell["code"] = prev_cell.get("code", "") + "\n" + cell.get("code", "")
                            cells_to_remove.append(prev_cell["id"])
                            st.rerun()
                
                # Show AI suggestions
                if f"cell_suggestion_{cell['id']}" in st.session_state:
                    with st.expander("💡 AI Suggestion", expanded=True):
                        st.code(st.session_state[f"cell_suggestion_{cell['id']}"], language="python")
                        if st.button("Use This", key=f"use_suggest_{cell['id']}"):
                            cell["code"] = st.session_state[f"cell_suggestion_{cell['id']}"]
                            del st.session_state[f"cell_suggestion_{cell['id']}"]
                            st.rerun()
                
                if f"cell_corrected_{cell['id']}" in st.session_state:
                    with st.expander("✅ Corrected Code", expanded=True):
                        st.code(st.session_state[f"cell_corrected_{cell['id']}"], language="python")
                        if st.button("Use This", key=f"use_corrected_{cell['id']}"):
                            cell["code"] = st.session_state[f"cell_corrected_{cell['id']}"]
                            del st.session_state[f"cell_corrected_{cell['id']}"]
                            st.rerun()
                
                if f"cell_completion_{cell['id']}" in st.session_state:
                    with st.expander("➕ Code Completion", expanded=True):
                        st.code(st.session_state[f"cell_completion_{cell['id']}"], language="python")
                        if st.button("Use This", key=f"use_complete_{cell['id']}"):
                            cell["code"] = st.session_state[f"cell_completion_{cell['id']}"]
                            del st.session_state[f"cell_completion_{cell['id']}"]
                            st.rerun()
            
            # Enhanced output display
            if cell.get("executed") and cell.get("cell_type", "code") == "code":
                if cell.get("error"):
                    with st.expander(f"❌ Error [{cell.get('execution_count', '?')}]", expanded=True):
                        # Enhanced error display with line numbers
                        error_lines = cell["error"].split('\n')
                        st.code(cell["error"], language="python")
                        # Try to extract line number
                        for line in error_lines[:5]:
                            if "line" in line.lower() and any(char.isdigit() for char in line):
                                st.warning(line)
                elif cell.get("output"):
                    with st.expander(f"📤 Output [{cell.get('execution_count', '?')}]", expanded=True):
                        output_text = cell["output"].strip()
                        # Check for HTML output
                        if output_text.startswith('<') and output_text.endswith('>'):
                            st.markdown(output_text, unsafe_allow_html=True)
                        else:
                            # Check for DataFrame display
                            display_output = True
                            for var_name, var_value in st.session_state.notebook_variables.items():
                                if isinstance(var_value, pd.DataFrame) and var_name in cell["code"]:
                                    st.dataframe(var_value, use_container_width=True)
                                    display_output = False
                                    break
                            
                            if display_output:
                                # Check if output contains image data
                                if "data:image" in output_text or "base64" in output_text.lower():
                                    try:
                                        # Extract and display image
                                        st.image(output_text)
                                    except:
                                        st.code(output_text, language="text")
                                else:
                                    st.code(output_text, language="text")
                else:
                    # Check for plots
                    plot_created = False
                    if MATPLOTLIB_AVAILABLE and plt:
                        for var_name, var_value in st.session_state.notebook_variables.items():
                            if 'fig' in var_name.lower():
                                try:
                                    if hasattr(var_value, 'savefig'):
                                        st.pyplot(var_value)
                                        plot_created = True
                                        break
                                except:
                                    pass
                        try:
                            if plt.get_fignums():
                                fig = plt.gcf()
                                if fig and len(fig.axes) > 0:
                                    st.pyplot(fig)
                                    plot_created = True
                                    plt.close('all')
                        except:
                            pass
                    
                    if not plot_created:
                        st.success(f"✓ Executed [{cell.get('execution_count', '?')}] - No output")
            
            st.markdown("---")
    
    # Remove cells
    if cells_to_remove:
        st.session_state.notebook_cells = [
            cell for cell in st.session_state.notebook_cells 
            if cell["id"] not in cells_to_remove
        ]
        st.rerun()
    
    # Enhanced Variables Inspector
    with st.sidebar.expander("📊 Variable Inspector", expanded=False):
        if st.session_state.notebook_variables:
            search_var = st.text_input("🔍 Search variables", "")
            vars_to_show = st.session_state.notebook_variables
            if search_var:
                vars_to_show = {k: v for k, v in vars_to_show.items() if search_var.lower() in k.lower()}
            
            for var_name, var_value in list(vars_to_show.items())[:30]:
                with st.expander(f"{var_name}", expanded=False):
                    var_type = type(var_value).__name__
                    st.text(f"Type: {var_type}")
                    try:
                        if isinstance(var_value, pd.DataFrame):
                            st.dataframe(var_value.head(), use_container_width=True)
                            st.text(f"Shape: {var_value.shape}")
                        elif isinstance(var_value, (list, tuple)):
                            st.text(f"Length: {len(var_value)}")
                            if len(var_value) < 20:
                                st.code(str(var_value))
                        elif isinstance(var_value, dict):
                            st.text(f"Keys: {len(var_value)}")
                            if len(var_value) < 10:
                                st.json(var_value)
                        else:
                            var_str = str(var_value)
                            if len(var_str) > 200:
                                var_str = var_str[:200] + "..."
                            st.code(var_str)
                    except:
                        st.text("<unable to display>")
        else:
            st.info("No variables defined yet")
        
        if st.button("🗑️ Clear All Variables"):
            st.session_state.notebook_variables = {}
            st.rerun()
    
    # Magic commands help
    with st.sidebar.expander("🔮 Magic Commands", expanded=False):
        st.markdown("""
        **Line Magics:**
        - `%time` - Time execution
        - `%timeit` - Time with multiple runs
        - `%matplotlib inline` - Set matplotlib backend
        - `%load <file>` - Load code from file
        - `%who` - List variables
        - `%whos` - Detailed variable info
        
        **Cell Magics:**
        - `%%time` - Time cell execution
        - `%%writefile <file>` - Write output to file
        """)

# AI Chat Page
elif page == "AI Chat":
    st.title("💬 AI Chat Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Build conversation context
                messages = []
                for msg in st.session_state.chat_history[-5:]:  # Last 5 messages for context
                    messages.append(f"{msg['role'].title()}: {msg['content']}")
                
                context = "\n".join(messages) if messages else ""
                full_prompt = f"{context}\nUser: {prompt}\nAssistant:" if context else prompt
                
                response = call_ollama(full_prompt, use_chat=True)
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Code Generator Page
elif page == "Code Generator":
    st.title("💻 AI Code Generator")
    
    description = st.text_area(
        "Describe what code you want to generate:",
        height=150,
        placeholder="e.g., 'Create a function that calculates the factorial of a number'"
    )
    
    language = st.selectbox("Programming Language", ["Python", "JavaScript", "Java", "C++", "Other"])
    
    if st.button("🚀 Generate Code"):
        if description:
            with st.spinner("Generating code..."):
                full_prompt = f"Generate {language} code: {description}"
                generated_code = generate_code(full_prompt)
                
                st.markdown("### Generated Code:")
                st.code(generated_code, language="python" if language == "Python" else "text")
                
                # Save option
                filename = st.text_input("Save as:", value="generated_code.py")
                if st.button("💾 Save Generated Code"):
                    filepath = os.path.join(st.session_state.workspace_path, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(generated_code)
                    st.success(f"Code saved to {filepath}")

# Visualization Generator Page
elif page == "Visualization Generator":
    st.title("📊 Visualization Generator")
    
    tab1, tab2 = st.tabs(["Generate from Description", "Upload Data"])
    
    with tab1:
        viz_description = st.text_area(
            "Describe the visualization you want:",
            height=100,
            placeholder="e.g., 'Create a bar chart showing sales by month'"
        )
        
        data_type = st.selectbox("Data Type", ["Sample Data", "CSV File", "JSON File", "Database"])
        
        if st.button("🎨 Generate Visualization"):
            if viz_description:
                with st.spinner("Generating visualization code..."):
                    viz_code = generate_visualization(viz_description, data_type)
                    
                    st.markdown("### Generated Visualization Code:")
                    st.code(viz_code, language="python")
                    
                    # Try to execute the code
                    try:
                        exec_globals = {"pd": pd, "px": px, "go": go, "st": st, "fig": None}
                        exec(viz_code, exec_globals)
                        
                        # Check if a figure was created
                        if 'fig' in exec_globals and exec_globals['fig'] is not None:
                            fig = exec_globals['fig']
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download buttons
                            st.markdown("### 📥 Download Visualization")
                            col_dl1, col_dl2, col_dl3 = st.columns(3)
                            
                            with col_dl1:
                                try:
                                    img_bytes = fig.to_image(format="png", width=1200, height=800)
                                    st.download_button(
                                        label="📷 Download PNG",
                                        data=img_bytes,
                                        file_name=f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                        mime="image/png"
                                    )
                                except Exception as e:
                                    st.warning(f"PNG: {str(e)}")
                            
                            with col_dl2:
                                try:
                                    html_str = fig.to_html()
                                    st.download_button(
                                        label="🌐 Download HTML",
                                        data=html_str,
                                        file_name=f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                        mime="text/html"
                                    )
                                except Exception as e:
                                    st.warning(f"HTML: {str(e)}")
                            
                            with col_dl3:
                                try:
                                    json_str = fig.to_json()
                                    st.download_button(
                                        label="📄 Download JSON",
                                        data=json_str,
                                        file_name=f"viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                        mime="application/json"
                                    )
                                except Exception as e:
                                    st.warning(f"JSON: {str(e)}")
                            
                            st.session_state.last_figure = fig
                        
                        st.success("Visualization generated successfully!")
                    except Exception as e:
                        st.warning(f"Code generated but needs adjustment: {str(e)}")
                        st.code(viz_code, language="python")
    
    with tab2:
        uploaded_file = st.file_uploader("Upload data file", type=["csv", "json", "xlsx"])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", df.columns)
                with col2:
                    y_col = st.selectbox("Y-axis", df.columns)
                
                chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie"])
                
                if st.button("Create Chart"):
                    if chart_type == "Bar":
                        fig = px.bar(df, x=x_col, y=y_col, title=f"{chart_type} Chart: {y_col} by {x_col}")
                    elif chart_type == "Line":
                        fig = px.line(df, x=x_col, y=y_col, title=f"{chart_type} Chart: {y_col} over {x_col}")
                    elif chart_type == "Scatter":
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"{chart_type} Chart: {y_col} vs {x_col}")
                    else:
                        fig = px.pie(df, names=x_col, values=y_col, title=f"{chart_type} Chart: {y_col} by {x_col}")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download buttons
                    st.markdown("### 📥 Download Visualization")
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        try:
                            img_bytes = fig.to_image(format="png", width=1200, height=800)
                            st.download_button(
                                label="📷 Download PNG",
                                data=img_bytes,
                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                        except Exception as e:
                            st.error(f"PNG export failed. Install kaleido: pip install kaleido")
                    
                    with col_dl2:
                        try:
                            html_str = fig.to_html()
                            st.download_button(
                                label="🌐 Download HTML",
                                data=html_str,
                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                mime="text/html"
                            )
                        except Exception as e:
                            st.error(f"HTML export failed: {str(e)}")
                    
                    with col_dl3:
                        try:
                            # Save as JSON (plotly format)
                            json_str = fig.to_json()
                            st.download_button(
                                label="📄 Download JSON",
                                data=json_str,
                                file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"JSON export failed: {str(e)}")
                    
                    # Store figure in session state for reuse
                    st.session_state.last_figure = fig

# Presentation Generator Page
elif page == "Presentation Generator":
    st.title("📊 AI Presentation Generator")
    st.markdown("### Create Professional Business Presentations with AI-Powered Visualizations")
    
    tab1, tab2 = st.tabs(["📈 Generate from Data", "📄 Generate from Description"])
    
    with tab1:
        st.markdown("#### Upload Your Data")
        uploaded_data = st.file_uploader("Upload data file", type=["csv", "json", "xlsx"], key="pres_data")
        
        if uploaded_data:
            try:
                if uploaded_data.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_data)
                elif uploaded_data.name.endswith('.json'):
                    df = pd.read_json(uploaded_data)
                elif uploaded_data.name.endswith('.xlsx'):
                    try:
                        df = pd.read_excel(uploaded_data)
                    except:
                        st.error("Excel file reading requires openpyxl. Install with: pip install openpyxl")
                        df = None
                else:
                    df = None
                
                if df is not None:
                    st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
                    with st.expander("📋 Preview Data", expanded=False):
                        st.dataframe(df.head(10))
                    
                    # Presentation settings
                    col_set1, col_set2 = st.columns(2)
                    with col_set1:
                        presentation_type = st.selectbox(
                            "Presentation Type",
                            ["Business", "Data Analysis", "Executive Summary", "Technical Report", "Sales Report"],
                            help="Choose the style of presentation"
                        )
                    
                    with col_set2:
                        viz_mode = st.radio(
                            "Visualization Selection",
                            ["🤖 AI Auto-Select", "👤 Manual Selection"],
                            help="Let AI choose best visualizations or select manually"
                        )
                    
                    # Manual visualization selection
                    selected_viz_types = []
                    if viz_mode == "👤 Manual Selection":
                        st.markdown("#### Select Visualizations")
                        viz_options = st.multiselect(
                            "Choose visualization types:",
                            ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Heatmap", 
                             "Box Plot", "Histogram", "Area Chart", "Bubble Chart"],
                            default=["Bar Chart", "Line Chart"],
                            help="Select which types of charts to include"
                        )
                        selected_viz_types = viz_options
                    
                    # Data description
                    data_description = st.text_area(
                        "Describe what you want to present:",
                        height=100,
                        placeholder="e.g., 'Analyze sales trends, identify top products, and provide recommendations for Q4 strategy'",
                        help="Tell AI what insights you want in the presentation"
                    )
                    
                    if st.button("🚀 Generate Presentation", type="primary", use_container_width=True):
                        if data_description:
                            with st.spinner("🤖 AI is creating your presentation... This may take a minute."):
                                presentation_code = generate_presentation(
                                    data_description, 
                                    data=df,
                                    presentation_type=presentation_type.lower(),
                                    auto_visualize=(viz_mode == "🤖 AI Auto-Select"),
                                    selected_viz=selected_viz_types
                                )
                                
                                st.markdown("### 📊 Generated Presentation")
                                
                                # Execute the presentation code
                                try:
                                    exec_globals = {
                                        "pd": pd, "px": px, "go": go, "st": st,
                                        "df": df, "plt": plt if MATPLOTLIB_AVAILABLE else None,
                                        "datetime": datetime
                                    }
                                    
                                    # Create a container for the presentation
                                    with st.container():
                                        exec(presentation_code, exec_globals)
                                    
                                    # Download presentation if HTML was created
                                    if 'presentation_html' in exec_globals:
                                        st.markdown("### 📥 Download Presentation")
                                        st.download_button(
                                            label="💾 Download HTML Presentation",
                                            data=exec_globals['presentation_html'],
                                            file_name=f"presentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                            mime="text/html"
                                        )
                                    
                                    st.success("✅ Presentation generated successfully!")
                                    
                                except Exception as e:
                                    st.error(f"Error executing presentation: {str(e)}")
                                    st.markdown("### Generated Code (for debugging):")
                                    st.code(presentation_code, language="python")
                        else:
                            st.warning("Please describe what you want to present.")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
        else:
            st.info("👆 Upload a CSV, JSON, or Excel file to get started")
    
    with tab2:
        st.markdown("#### Generate Presentation from Description")
        
        presentation_desc = st.text_area(
            "Describe your presentation:",
            height=150,
            placeholder="e.g., 'Create a business presentation about quarterly sales performance. Include trends, top products, regional analysis, and strategic recommendations for the next quarter.'"
        )
        
        col_type1, col_type2 = st.columns(2)
        with col_type1:
            pres_type = st.selectbox(
                "Presentation Type",
                ["Business", "Data Analysis", "Executive Summary", "Technical Report", "Sales Report"],
                key="pres_type_desc"
            )
        
        with col_type2:
            auto_viz = st.checkbox("🤖 Let AI choose visualizations", value=True)
        
        if st.button("🚀 Generate Presentation", type="primary", use_container_width=True, key="gen_pres_desc"):
            if presentation_desc:
                with st.spinner("🤖 AI is creating your presentation..."):
                    presentation_code = generate_presentation(
                        presentation_desc,
                        data=None,
                        presentation_type=pres_type.lower(),
                        auto_visualize=auto_viz,
                        selected_viz=None
                    )
                    
                    st.markdown("### 📊 Generated Presentation Code")
                    st.code(presentation_code, language="python")
                    
                    # Try to execute
                    try:
                        exec_globals = {"pd": pd, "px": px, "go": go, "st": st, "datetime": datetime}
                        exec(presentation_code, exec_globals)
                        st.success("✅ Presentation code executed!")
                    except Exception as e:
                        st.warning(f"Code generated. Adjust as needed: {str(e)}")
            else:
                st.warning("Please describe your presentation.")

# App Generator Page
elif page == "App Generator":
    st.title("🎨 Streamlit App Generator")
    
    app_description = st.text_area(
        "Describe the app you want to create:",
        height=200,
        placeholder="e.g., 'Create a todo list app with add, delete, and mark complete features'"
    )
    
    if st.button("🚀 Generate App"):
        if app_description:
            with st.spinner("Generating your app..."):
                app_code = generate_app(app_description)
                
                st.markdown("### Generated App Code:")
                st.code(app_code, language="python")
                
                # Save option
                filename = st.text_input("Save as:", value="generated_app.py")
                if st.button("💾 Save App"):
                    filepath = os.path.join(st.session_state.workspace_path, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(app_code)
                    st.success(f"App saved to {filepath}")
                    st.info(f"Run with: streamlit run {filepath}")

# File Manager Page
elif page == "File Manager":
    st.title("📁 File Manager")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Current Workspace Files")
        workspace_files = []
        if os.path.exists(st.session_state.workspace_path):
            for root, dirs, files in os.walk(st.session_state.workspace_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), st.session_state.workspace_path)
                    workspace_files.append(rel_path)
        
        if workspace_files:
            selected_file = st.selectbox("Select a file:", workspace_files)
            if selected_file:
                filepath = os.path.join(st.session_state.workspace_path, selected_file)
                if st.button("📖 View File"):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                        st.code(content, language="python")
                    except:
                        st.error("Could not read file")
                
                if st.button("🗑️ Delete File"):
                    os.remove(filepath)
                    st.success("File deleted!")
                    st.rerun()
        else:
            st.info("No files in workspace yet")
    
    with col2:
        st.markdown("### Create New")
        
        # Create folder
        folder_name = st.text_input("Folder Name:")
        if st.button("📂 Create Folder"):
            folder_path = os.path.join(st.session_state.workspace_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            st.success(f"Folder created: {folder_path}")
        
        st.markdown("---")
        
        # Save data
        st.markdown("### Save Data")
        data_type = st.selectbox("Data Type", ["JSON", "CSV", "Text"])
        data_name = st.text_input("File Name:")
        
        if data_type == "JSON":
            json_data = st.text_area("JSON Data:", height=100)
            if st.button("💾 Save JSON"):
                if json_data and data_name:
                    try:
                        data = json.loads(json_data)
                        filepath = os.path.join(st.session_state.workspace_path, f"{data_name}.json")
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                        st.success(f"Saved to {filepath}")
                    except:
                        st.error("Invalid JSON")
        
        elif data_type == "CSV":
            csv_data = st.text_area("CSV Data (comma-separated):", height=100)
            if st.button("💾 Save CSV"):
                if csv_data and data_name:
                    filepath = os.path.join(st.session_state.workspace_path, f"{data_name}.csv")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(csv_data)
                    st.success(f"Saved to {filepath}")
        
        else:
            text_data = st.text_area("Text Data:", height=100)
            if st.button("💾 Save Text"):
                if text_data and data_name:
                    filepath = os.path.join(st.session_state.workspace_path, f"{data_name}.txt")
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(text_data)
                    st.success(f"Saved to {filepath}")

# Workspace Page
elif page == "Workspace":
    st.title("📂 Workspace Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Current Workspace")
        st.info(f"Path: {st.session_state.workspace_path}")
        
        # Change workspace
        new_workspace = st.text_input("Change Workspace Path:")
        if st.button("📂 Change Workspace"):
            if new_workspace:
                os.makedirs(new_workspace, exist_ok=True)
                st.session_state.workspace_path = new_workspace
                st.success(f"Workspace changed to {new_workspace}")
                st.rerun()
    
    with col2:
        st.markdown("### Workspace Stats")
        if os.path.exists(st.session_state.workspace_path):
            files = []
            folders = []
            for item in os.listdir(st.session_state.workspace_path):
                item_path = os.path.join(st.session_state.workspace_path, item)
                if os.path.isfile(item_path):
                    files.append(item)
                else:
                    folders.append(item)
            
            st.metric("Files", len(files))
            st.metric("Folders", len(folders))
            st.metric("Total Items", len(files) + len(folders))
    
    st.markdown("---")
    st.markdown("### Workspace Explorer")
    
    # Tree view
    def show_tree(path, level=0):
        indent = "  " * level
        try:
            items = sorted(os.listdir(path))
            for item in items:
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    st.text(f"{indent}📁 {item}/")
                    if st.checkbox(f"Expand {item}", key=f"expand_{item_path}"):
                        show_tree(item_path, level + 1)
                else:
                    size = os.path.getsize(item_path)
                    st.text(f"{indent}📄 {item} ({size} bytes)")
        except:
            pass
    
    if os.path.exists(st.session_state.workspace_path):
        show_tree(st.session_state.workspace_path)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
ollama_url = st.sidebar.text_input("Ollama URL", value=OLLAMA_BASE_URL)
model_name = st.sidebar.text_input("Model Name", value=DEFAULT_MODEL)

if st.sidebar.button("🔄 Test Connection"):
    test_response = call_ollama("Hello", model=model_name)
    if "Error" not in test_response:
        st.sidebar.success("✓ Connected to Ollama")
    else:
        st.sidebar.error("✗ Connection failed")

# Footer with creator info
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{APP_NAME}**")
st.sidebar.markdown(f"Version: {APP_VERSION}")
st.sidebar.markdown(f"Created by: **{APP_CREATOR}**")
if icon_path and os.path.exists(icon_path):
    st.sidebar.image(icon_path, width=32)

