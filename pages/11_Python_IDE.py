"""
Python IDE - Full-Featured Code Editor with AI Assistant
Created by: Mohammad Saeed Angiz
"""

import streamlit as st
import os
import sys
import io
import ast
import contextlib
import subprocess
import traceback
from datetime import datetime

# Optional richer editor
try:
    from streamlit_ace import st_ace
    ACE_AVAILABLE = True
except ImportError:
    ACE_AVAILABLE = False
    st_ace = None

st.set_page_config(page_title="Python IDE", page_icon="💻", layout="wide")

# ---------- Styles ----------
st.markdown("""
<style>
    .ide-topbar {
        background:#1e1e1e;color:#eee;padding:.5rem 1rem;border-radius:6px;
        display:flex;gap:1rem;align-items:center;margin-bottom:.5rem;font-family:monospace;
    }
    .tab-active {background:#2d2d2d;color:#fff;padding:.25rem .75rem;border-radius:4px 4px 0 0;}
    .tab-inactive {background:#3c3c3c;color:#bbb;padding:.25rem .75rem;border-radius:4px 4px 0 0;}
    .output-box {
        background:#0c0c0c;color:#d4d4d4;padding:1rem;border-radius:6px;
        font-family:'Consolas','Courier New',monospace;font-size:.85rem;
        white-space:pre-wrap;min-height:180px;max-height:420px;overflow:auto;
    }
    .status-bar {
        background:#007acc;color:white;padding:.25rem 1rem;border-radius:4px;
        font-family:monospace;font-size:.8rem;margin-top:.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session state ----------
DEFAULTS = {
    'ide_files': {'main.py': '# Welcome to AI Lab Super Python IDE\n# Created by Mohammad Saeed Angiz\n\nprint("Hello, World!")\n'},
    'ide_current_file': 'main.py',
    'ide_open_files': ['main.py'],
    'ide_output': '',
    'ide_terminal_history': [],
    'ide_globals': {},                # persistent exec namespace
    'ide_theme': 'monokai',
    'ide_font_size': 14,
    'ide_tab_size': 4,
    'ide_wrap': False,
    'ide_last_run': None,
    'ai_provider': 'OpenAI',
    'ai_model': 'gpt-4',
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


# ---------- Core IDE functions (defined BEFORE they are used) ----------
def execute_python_code(code: str) -> dict:
    """Execute Python code in a persistent namespace, capture stdout/stderr."""
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    start = datetime.now()
    result = {'stdout': '', 'stderr': '', 'error': None, 'duration': 0.0}
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(compile(code, st.session_state.ide_current_file or '<ide>', 'exec'),
                 st.session_state.ide_globals)
    except Exception:
        result['error'] = traceback.format_exc()
    result['stdout'] = stdout_buf.getvalue()
    result['stderr'] = stderr_buf.getvalue()
    result['duration'] = (datetime.now() - start).total_seconds()
    return result


def analyze_code(code: str) -> dict:
    """Static analysis via AST: count functions, classes, imports, syntax errors."""
    info = {'valid': True, 'error': None, 'functions': [], 'classes': [],
            'imports': [], 'lines': len(code.splitlines()), 'chars': len(code)}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                info['functions'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                info['classes'].append(node.name)
            elif isinstance(node, ast.Import):
                info['imports'].extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                info['imports'].append(node.module or '')
    except SyntaxError as e:
        info['valid'] = False
        info['error'] = f"Line {e.lineno}: {e.msg}"
    return info


def lint_code(code: str) -> list:
    """Lightweight lint using ast + heuristic checks (no external deps)."""
    issues = []
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"[SyntaxError] Line {e.lineno}: {e.msg}")
        return issues
    for i, line in enumerate(code.splitlines(), 1):
        if len(line) > 120:
            issues.append(f"[E501] Line {i}: line too long ({len(line)} > 120)")
        if line.rstrip() != line and line.strip():
            issues.append(f"[W291] Line {i}: trailing whitespace")
        if '\t' in line:
            issues.append(f"[W191] Line {i}: indentation contains tabs")
    if not issues:
        issues.append("✅ No issues found.")
    return issues


def execute_terminal_command(cmd: str) -> str:
    """Run a shell command safely (short timeout) and return combined output."""
    if not cmd.strip():
        return ""
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=30)
        return (proc.stdout or '') + (proc.stderr or '')
    except subprocess.TimeoutExpired:
        return "[Error] Command timed out after 30s"
    except Exception as e:
        return f"[Error] {e}"


# ---------- AI assistant stubs (safe placeholders; wire real API here) ----------
def _ai_notice(action: str, detail: str = "") -> str:
    provider = st.session_state.get('ai_provider', 'OpenAI')
    model = st.session_state.get('ai_model', 'gpt-4')
    header = f"# [AI:{provider}/{model}] {action}"
    body = f"# (Configure your API key in the sidebar to enable real generation)\n# {detail}" if detail else ""
    return f"{header}\n{body}\n"


def ai_generate_code(prompt: str, language: str = "python") -> str:
    return _ai_notice("generate_code", prompt) + f"# TODO: implement: {prompt}\npass\n"


def ai_complete_code(code: str) -> str:
    return code + "\n" + _ai_notice("complete_code", "completion placeholder")


def ai_find_bugs(code: str) -> str:
    analysis = analyze_code(code)
    if not analysis['valid']:
        return f"Syntax error detected: {analysis['error']}"
    return _ai_notice("find_bugs", f"{len(code.splitlines())} lines scanned") + "No obvious static issues."


def ai_add_documentation(code: str) -> str:
    return _ai_notice("add_documentation") + code


def ai_optimize_code(code: str) -> str:
    return _ai_notice("optimize_code") + code


def ai_generate_tests(code: str) -> str:
    analysis = analyze_code(code)
    tests = _ai_notice("generate_tests")
    tests += "import pytest\n\n"
    for fn in analysis['functions']:
        tests += f"def test_{fn}():\n    # TODO: add assertions\n    assert {fn} is not None\n\n"
    if not analysis['functions']:
        tests += "def test_placeholder():\n    assert True\n"
    return tests


# ---------- Helpers ----------
def render_editor(code: str, key: str) -> str:
    if ACE_AVAILABLE:
        return st_ace(
            value=code,
            language="python",
            theme=st.session_state.ide_theme,
            keybinding="vscode",
            font_size=st.session_state.ide_font_size,
            tab_size=st.session_state.ide_tab_size,
            wrap=st.session_state.ide_wrap,
            show_gutter=True,
            show_print_margin=True,
            auto_update=False,
            min_lines=24,
            max_lines=40,
            key=key,
        )
    return st.text_area("Editor", value=code, height=500, key=key,
                        label_visibility="collapsed")


def open_file(name: str):
    if name not in st.session_state.ide_open_files:
        st.session_state.ide_open_files.append(name)
    st.session_state.ide_current_file = name


def close_file(name: str):
    if name in st.session_state.ide_open_files:
        st.session_state.ide_open_files.remove(name)
    if st.session_state.ide_current_file == name:
        st.session_state.ide_current_file = (
            st.session_state.ide_open_files[-1] if st.session_state.ide_open_files else None
        )


# ---------- UI ----------
st.markdown("# 💻 Python IDE")
st.caption("Created by: Mohammad Saeed Angiz")

if not ACE_AVAILABLE:
    st.info("Install **streamlit-ace** for syntax highlighting: `pip install streamlit-ace`")

# Top toolbar
tb = st.columns([1, 1, 1, 1, 1, 1, 2])
with tb[0]:
    if st.button("▶ Run", use_container_width=True, type="primary"):
        st.session_state._run_now = True
with tb[1]:
    if st.button("💾 Save", use_container_width=True):
        st.toast(f"Saved {st.session_state.ide_current_file}")
with tb[2]:
    if st.button("🧹 Clear Out", use_container_width=True):
        st.session_state.ide_output = ''
with tb[3]:
    if st.button("🔄 Reset Kernel", use_container_width=True):
        st.session_state.ide_globals = {}
        st.toast("Kernel reset")
with tb[4]:
    if st.button("🐛 Lint", use_container_width=True):
        st.session_state._lint_now = True
with tb[5]:
    if st.button("📊 Analyze", use_container_width=True):
        st.session_state._analyze_now = True

# Main layout: explorer | editor+output | AI
col_explorer, col_main, col_ai = st.columns([1, 3, 1.2])

# ----- File Explorer -----
with col_explorer:
    st.markdown("### 📁 Files")
    new_name = st.text_input("New file name", placeholder="script.py", key="new_file_name")
    if st.button("➕ Create", use_container_width=True):
        if new_name and new_name not in st.session_state.ide_files:
            st.session_state.ide_files[new_name] = f"# {new_name}\n"
            open_file(new_name)
            st.rerun()

    uploaded = st.file_uploader("Upload .py", type=['py', 'txt'], key="py_uploader")
    if uploaded is not None:
        st.session_state.ide_files[uploaded.name] = uploaded.read().decode('utf-8', errors='replace')
        open_file(uploaded.name)

    st.markdown("---")
    for fname in sorted(st.session_state.ide_files.keys()):
        c1, c2 = st.columns([4, 1])
        if c1.button(f"📄 {fname}", key=f"open_{fname}", use_container_width=True):
            open_file(fname)
            st.rerun()
        if c2.button("✕", key=f"del_{fname}"):
            st.session_state.ide_files.pop(fname, None)
            close_file(fname)
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.session_state.ide_theme = st.selectbox(
        "Theme", ["monokai", "github", "tomorrow", "twilight", "solarized_dark",
                  "solarized_light", "dracula", "xcode"],
        index=0)
    st.session_state.ide_font_size = st.slider("Font size", 10, 24, 14)
    st.session_state.ide_tab_size = st.slider("Tab size", 2, 8, 4)
    st.session_state.ide_wrap = st.checkbox("Word wrap", value=False)

# ----- Editor + Output -----
with col_main:
    # Open tabs
    if st.session_state.ide_open_files:
        tab_cols = st.columns(len(st.session_state.ide_open_files) + 1)
        for i, fname in enumerate(st.session_state.ide_open_files):
            with tab_cols[i]:
                label = f"● {fname}" if fname == st.session_state.ide_current_file else fname
                if st.button(label, key=f"tab_{fname}", use_container_width=True):
                    st.session_state.ide_current_file = fname
                    st.rerun()
        with tab_cols[-1]:
            if st.session_state.ide_current_file and st.button("✕ close", key="close_tab"):
                close_file(st.session_state.ide_current_file)
                st.rerun()

    cur = st.session_state.ide_current_file
    if cur:
        code = render_editor(st.session_state.ide_files[cur], key=f"editor_{cur}")
        if code is not None:
            st.session_state.ide_files[cur] = code

        # Status bar
        lines = len(st.session_state.ide_files[cur].splitlines())
        chars = len(st.session_state.ide_files[cur])
        st.markdown(
            f"<div class='status-bar'>📄 {cur} | Lines: {lines} | Chars: {chars} | "
            f"Theme: {st.session_state.ide_theme} | Python {sys.version.split()[0]}</div>",
            unsafe_allow_html=True,
        )

        # Run / Lint / Analyze actions
        if st.session_state.pop('_run_now', False):
            res = execute_python_code(st.session_state.ide_files[cur])
            out = ''
            if res['stdout']:
                out += res['stdout']
            if res['stderr']:
                out += "\n[stderr]\n" + res['stderr']
            if res['error']:
                out += "\n[Traceback]\n" + res['error']
            out += f"\n--- finished in {res['duration']:.3f}s ---"
            st.session_state.ide_output = out
            st.session_state.ide_last_run = datetime.now().strftime("%H:%M:%S")

        if st.session_state.pop('_lint_now', False):
            st.session_state.ide_output = "\n".join(lint_code(st.session_state.ide_files[cur]))

        if st.session_state.pop('_analyze_now', False):
            info = analyze_code(st.session_state.ide_files[cur])
            st.session_state.ide_output = (
                f"Valid: {info['valid']}\n"
                f"Error: {info['error']}\n"
                f"Lines: {info['lines']}   Chars: {info['chars']}\n"
                f"Functions ({len(info['functions'])}): {', '.join(info['functions']) or '-'}\n"
                f"Classes   ({len(info['classes'])}): {', '.join(info['classes']) or '-'}\n"
                f"Imports   ({len(info['imports'])}): {', '.join(info['imports']) or '-'}"
            )
    else:
        st.info("No file open. Create or upload one from the File Explorer.")

    # Output / Terminal tabs
    st.markdown("### 📤 Output")
    out_tab, term_tab = st.tabs(["Output", "Terminal"])
    with out_tab:
        st.markdown(f"<div class='output-box'>{(st.session_state.ide_output or '&nbsp;')}</div>",
                    unsafe_allow_html=True)
        if st.session_state.ide_last_run:
            st.caption(f"Last run: {st.session_state.ide_last_run}")

    with term_tab:
        cmd = st.text_input("$", key="term_cmd", placeholder="e.g. pip list")
        c1, c2 = st.columns([1, 1])
        if c1.button("Execute", key="term_exec"):
            if cmd:
                out = execute_terminal_command(cmd)
                st.session_state.ide_terminal_history.append(f"$ {cmd}\n{out}")
        if c2.button("Clear terminal", key="term_clear"):
            st.session_state.ide_terminal_history = []
        history = "\n".join(st.session_state.ide_terminal_history[-50:])
        st.markdown(f"<div class='output-box'>{history or '&nbsp;'}</div>",
                    unsafe_allow_html=True)

# ----- AI Panel -----
with col_ai:
    st.markdown("### 🤖 AI Assistant")
    st.caption(f"{st.session_state.ai_provider} / {st.session_state.ai_model}")

    prompt = st.text_area("Describe what to generate:", height=100,
                          placeholder="e.g. function to compute fibonacci",
                          key="ai_prompt")
    if st.button("✨ Generate", use_container_width=True):
        code = ai_generate_code(prompt)
        cur = st.session_state.ide_current_file
        if cur:
            st.session_state.ide_files[cur] += "\n" + code
            st.rerun()

    st.markdown("---")
    cur = st.session_state.ide_current_file
    current_code = st.session_state.ide_files.get(cur, "") if cur else ""

    if st.button("🧩 Complete code", use_container_width=True) and cur:
        st.session_state.ide_files[cur] = ai_complete_code(current_code)
        st.rerun()

    if st.button("🐞 Find bugs", use_container_width=True) and cur:
        st.session_state.ide_output = ai_find_bugs(current_code)

    if st.button("📝 Add docs", use_container_width=True) and cur:
        st.session_state.ide_files[cur] = ai_add_documentation(current_code)
        st.rerun()

    if st.button("⚡ Optimize", use_container_width=True) and cur:
        st.session_state.ide_files[cur] = ai_optimize_code(current_code)
        st.rerun()

    if st.button("🧪 Generate tests", use_container_width=True) and cur:
        tname = (cur.rsplit('.', 1)[0] if cur else 'module') + "_test.py"
        st.session_state.ide_files[tname] = ai_generate_tests(current_code)
        open_file(tname)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Download")
    if cur:
        st.download_button("Download current file",
                           data=st.session_state.ide_files[cur],
                           file_name=cur, mime="text/x-python",
                           use_container_width=True)