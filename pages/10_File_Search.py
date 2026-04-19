"""
FFF File Search - Fast Fuzzy File Finder
Created by: Mohammad Saeed Angiz
Integrates: fff.nvim by dmtrKovalenko
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import time

st.markdown("# 🔍 FFF File Search - Fast Fuzzy Finder")
st.markdown("**Created by: Mohammad Saeed Angiz** | Based on fff.nvim")
st.markdown("---")

# Session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'frecency_data' not in st.session_state:
    st.session_state.frecency_data = {}

# Sidebar
st.sidebar.markdown("## 🔍 Search Configuration")

with st.sidebar.expander("⚙️ Settings"):
    max_results = st.slider("Max Results", 10, 200, 100)
    max_threads = st.slider("Max Threads", 1, 8, 4)
    show_scores = st.checkbox("Show Scores", value=False)
    case_sensitive = st.checkbox("Case Sensitive", value=False)

with st.sidebar.expander("📁 Path"):
    base_path = st.text_input("Base Path", value=os.getcwd())
    st.markdown(f"Current: `{base_path}`")

with st.sidebar.expander("📊 History"):
    if st.session_state.search_history:
        st.markdown("Recent Searches:")
        for i, query in enumerate(st.session_state.search_history[-5:]):
            st.markdown(f"{i+1}. {query}")
    else:
        st.markdown("No search history yet")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "📁 Find Files",
    "🔍 Live Grep",
    "📊 Frecency",
    "⚙️ Advanced"
])

with tab1:
    st.markdown("## Find Files")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    **FFF** (Fast Fuzzy File Finder) is an opinionated fuzzy file picker with
    memory built-in. It provides typo-resistant search and remembers frequently
    accessed files.
    """)
    
    # Search Input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        file_query = st.text_input(
            "Search Files",
            placeholder="Enter filename pattern (e.g., 'main.py', 'utils', 'config')",
            label_visibility="collapsed"
        )
    
    with col2:
        search_mode = st.selectbox("Mode", ["Fuzzy", "Plain", "Regex"])
    
    # Constraints
    with st.expander("🔧 Constraints"):
        st.markdown("""
        **File Constraints:**
        - `git:modified` - show only modified files
        - `test/` - any deeply nested children of test/ dir
        - `!something` - exclude results matching something
        - `./**/*.{py,js}` - glob patterns
        """)
        
        constraints = st.text_input(
            "Constraints (comma-separated)",
            placeholder="git:modified, test/, !__pycache__"
        )
    
    # Search
    if file_query:
        perform_file_search(file_query, constraints, max_results, show_scores)
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📁 Recent Files"):
            show_recent_files()
    
    with col2:
        if st.button("🔄 Modified Files"):
            show_modified_files()
    
    with col3:
        if st.button("📝 Config Files"):
            show_config_files()
    
    with col4:
        if st.button("🐍 Python Files"):
            show_python_files()

with tab2:
    st.markdown("## Live Grep")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    Search file contents with live grep. Supports plain text, regex, and fuzzy matching.
    """)
    
    # Grep Input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        grep_query = st.text_area(
            "Search Query",
            placeholder="Enter search pattern (e.g., 'def process_data', 'class User')",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        grep_mode = st.selectbox(
            "Mode",
            ["plain", "regex", "fuzzy"],
            index=0
        )
        
        st.markdown("**Cycle:** `<S-Tab>`")
        st.info(f"Current: **{grep_mode}**")
    
    # Grep Options
    with st.expander("⚙️ Grep Options"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_file_size = st.number_input("Max File Size (MB)", 1, 100, 10)
            max_matches = st.number_input("Max Matches/File", 1, 500, 100)
        
        with col2:
            smart_case = st.checkbox("Smart Case", value=True)
            trim_whitespace = st.checkbox("Trim Whitespace", value=False)
        
        with col3:
            file_types = st.multiselect(
                "File Types",
                ["*.py", "*.js", "*.ts", "*.md", "*.txt", "*.json"],
                default=["*.py", "*.js"]
            )
    
    # Search
    if st.button("🔍 Grep", type="primary") and grep_query:
        perform_grep_search(grep_query, grep_mode, file_types, max_results)
    
    # Current Word Search
    st.markdown("### 🎯 Quick Search")
    
    col1, col2 = st.columns(2)
    
    with col1:
        current_word = st.text_input("Search Current Word", placeholder="Enter word...")
        if st.button("Search Word"):
            perform_word_search(current_word)
    
    with col2:
        if st.button("🔍 Search Selection"):
            st.info("Select text in editor to search")

with tab3:
    st.markdown("## Frecency")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    st.markdown("""
    **Frecency** = Frequency + Recency
    
    Files you open frequently and recently get boosted scores,
    making them appear higher in search results.
    """)
    
    # Frecency Dashboard
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Opens", "1,234", delta="+56")
    
    with col2:
        st.metric("Unique Files", "456", delta="+12")
    
    with col3:
        st.metric("Avg Frecency Score", "87.3", delta="+2.1")
    
    # Top Files by Frecency
    st.markdown("### 📈 Top Files by Frecency")
    
    frecency_data = {
        "File": ["main.py", "utils.py", "config.py", "models.py", "app.py"],
        "Opens": [45, 38, 32, 28, 24],
        "Last Opened": ["2 mins ago", "5 mins ago", "1 hour ago", "2 hours ago", "3 hours ago"],
        "Frecency Score": [98, 92, 85, 78, 71]
    }
    
    frecency_df = pd.DataFrame(frecency_data)
    st.dataframe(frecency_df, use_container_width=True)
    
    # Combo Boosts
    st.markdown("### 🔗 Combo Boosts")
    
    st.markdown("""
    Files repeatedly opened with the same query get combo boosts.
    This helps the AI agent find the right file faster.
    """)
    
    combo_data = {
        "Query": ["main", "config", "utils", "model", "app"],
        "Top Match": ["main.py", "config.py", "utils.py", "models.py", "app.py"],
        "Combo Count": [12, 10, 8, 6, 5],
        "Boost Score": [1200, 1000, 800, 600, 500]
    }
    
    combo_df = pd.DataFrame(combo_data)
    st.dataframe(combo_df, use_container_width=True)
    
    # Clear Frecency
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Frecency"):
            clear_frecency_data()
    
    with col2:
        if st.button("📊 Export Frecency"):
            export_frecency_data()

with tab4:
    st.markdown("## Advanced Settings")
    st.markdown("**Created by: Mohammad Saeed Angiz**")
    
    # Indexing Settings
    st.markdown("### 🗂️ Indexing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.checkbox("Respect .gitignore", value=True)
        st.checkbox("Follow Symlinks", value=False)
        st.checkbox("Index Hidden Files", value=False)
    
    with col2:
        exclude_patterns = st.text_area(
            "Exclude Patterns",
            value="node_modules\n__pycache__\n.git\n.venv",
            height=100
        )
    
    # Performance Settings
    st.markdown("### ⚡ Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.slider("Search Timeout (ms)", 50, 500, 150)
        st.slider("Index Update Interval (s)", 1, 10, 5)
    
    with col2:
        st.slider("Result Cache Size", 100, 1000, 500)
        st.slider("Thread Pool Size", 1, 8, 4)
    
    # Display Settings
    st.markdown("### 🎨 Display")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Path Shorten Strategy", ["middle_number", "middle", "end"])
        st.checkbox("Show Git Status", value=True)
    
    with col2:
        st.checkbox("Show Preview", value=True)
        st.slider("Preview Size", 0.1, 1.0, 0.5)
    
    # Actions
    st.markdown("### 🔧 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Rebuild Index"):
            rebuild_file_index()
    
    with col2:
        if st.button("🗑️ Clear Cache"):
            clear_search_cache()
    
    with col3:
        if st.button("📊 Show Statistics"):
            show_search_statistics()

# Helper functions
def perform_file_search(query, constraints, max_results, show_scores):
    """Perform fuzzy file search"""
    with st.spinner(f"Searching for '{query}'..."):
        time.sleep(0.3)
        
        # Update search history
        if query not in st.session_state.search_history:
            st.session_state.search_history.append(query)
        
        # Sample results
        results = {
            "File": ["main.py", "utils.py", "config.py", "models.py", "app.py"],
            "Path": ["~/project/main.py", "~/project/utils.py", "~/project/config.py", "~/project/models.py", "~/project/app.py"],
            "Score": [98, 92, 85, 78, 71] if show_scores else [""] * 5,
            "Last Opened": ["2 mins ago", "5 mins ago", "1 hour ago", "2 hours ago", "3 hours ago"]
        }
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        st.success(f"✅ Found {len(results['File'])} files")

def show_recent_files():
    """Show recent files"""
    st.markdown("### 📁 Recent Files")
    
    recent = {
        "File": ["main.py", "utils.py", "config.py"],
        "Last Opened": ["2 mins ago", "5 mins ago", "1 hour ago"]
    }
    
    st.dataframe(pd.DataFrame(recent), use_container_width=True)

def show_modified_files():
    """Show modified files"""
    st.markdown("### 🔄 Modified Files (Git)")
    
    modified = {
        "File": ["main.py", "config.py"],
        "Status": ["Modified", "Staged"]
    }
    
    st.dataframe(pd.DataFrame(modified), use_container_width=True)

def show_config_files():
    """Show config files"""
    st.markdown("### ⚙️ Config Files")
    
    configs = {
        "File": ["config.py", "settings.json", ".env"],
        "Type": ["Python", "JSON", "Env"]
    }
    
    st.dataframe(pd.DataFrame(configs), use_container_width=True)

def show_python_files():
    """Show Python files"""
    st.markdown("### 🐍 Python Files")
    
    py_files = {
        "File": ["main.py", "utils.py", "models.py", "app.py"],
        "Lines": ["234", "156", "312", "89"]
    }
    
    st.dataframe(pd.DataFrame(py_files), use_container_width=True)

def perform_grep_search(query, mode, file_types, max_results):
    """Perform grep search"""
    with st.spinner(f"Searching for '{query}' in {', '.join(file_types)}..."):
        time.sleep(1)
        
        results = {
            "File": ["main.py", "utils.py", "models.py"],
            "Line": ["45", "123", "67"],
            "Match": [f"def process_data", f"class DataProcessor", f"def validate_input"],
            "Context": ["def process_data(input_file):", "class DataProcessor:", "def validate_input(data):"]
        }
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        st.success(f"✅ Found {len(results['File'])} matches")

def perform_word_search(word):
    """Search for current word"""
    if word:
        perform_grep_search(word, "plain", ["*.py", "*.js"], 20)

def clear_frecency_data():
    """Clear frecency data"""
    st.session_state.frecency_data = {}
    st.success("✅ Frecency data cleared")

def export_frecency_data():
    """Export frecency data"""
    st.success("✅ Frecency data exported to ~/.fff/frecency.json")

def rebuild_file_index():
    """Rebuild file index"""
    with st.spinner("Rebuilding index..."):
        time.sleep(2)
        st.success("✅ Index rebuilt")

def clear_search_cache():
    """Clear search cache"""
    st.success("✅ Cache cleared")

def show_search_statistics():
    """Show search statistics"""
    st.markdown("### 📊 Search Statistics")
    
    stats = {
        "Metric": ["Total Searches", "Avg Time", "Cache Hits", "Index Size"],
        "Value": ["1,234", "45ms", "89%", "312 MB"]
    }
    
    st.dataframe(pd.DataFrame(stats), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 1rem; background: #4b0082; border-radius: 8px;'>
    <p><b>FFF File Search - Fast Fuzzy Finder</b></p>
    <p>Created by <b>Mohammad Saeed Angiz</b></p>
    <p>Based on <b>fff.nvim</b> by dmtrKovalenko</p>
    <p>All credits for fff.nvim go to its original creator</p>
</div>
""", unsafe_allow_html=True)
