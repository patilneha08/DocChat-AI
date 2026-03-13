"""
Streamlit UI for the Document Q&A Chatbot.
Upload a document, ask questions, and get grounded answers.
"""
import streamlit as st
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ── Page config (MUST be first Streamlit call) ───────────────
st.set_page_config(page_title="DocChat AI", page_icon="📄", layout="wide")

# ── Dependency check ─────────────────────────────────────────
def check_dependencies():
    missing = []
    checks = {
        "langchain":            "langchain",
        "langchain_community":  "langchain-community",
        "langchain_chroma":     "langchain-chroma",
        "langchain_ollama":     "langchain-ollama",
        "langchain_huggingface":"langchain-huggingface",
        "chromadb":             "chromadb",
        "sentence_transformers":"sentence-transformers",
        "pypdf":                "pypdf",
    }
    for module, pip_name in checks.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    return missing

missing = check_dependencies()
if missing:
    st.title("📄 DocChat AI")
    st.error("**Missing dependencies detected!**")
    st.code(f"pip install {' '.join(missing)}", language="bash")
    st.info("Run the command above in your terminal, then refresh this page.")
    st.stop()

# ── Check Ollama is running ──────────────────────────────────
def check_ollama():
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False

if not check_ollama():
    st.title("📄 DocChat AI")
    st.error("**Ollama is not running!**")
    st.markdown(
        "1. Install Ollama from [ollama.com](https://ollama.com)\n"
        "2. Run `ollama serve` in a separate terminal\n"
        "3. Pull the model: `ollama pull mistral`\n"
        "4. Refresh this page"
    )
    st.stop()

# ── Now safe to import project modules ───────────────────────
import config
from ingest import ingest_file
from chain import build_chain
from config import sanitize_collection_name

# ── Polished CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Hide default Streamlit header bar and footer */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1);
    }

    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 8px;
    }
    [data-testid="stFileUploader"] label {
        font-weight: 600 !important;
    }

    /* Chat message bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    /* Welcome card */
    .welcome-card {
        text-align: center;
        padding: 60px 20px;
        max-width: 600px;
        margin: 0 auto;
    }
    .welcome-card h1 {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .welcome-card p {
        color: #888;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    /* Status pill in sidebar */
    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 4px 0;
    }
    .status-ready {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981 !important;
    }
    .status-waiting {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b !important;
    }

    /* Source expander */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        font-weight: 600;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
    }

    /* Suggested question buttons */
    .suggestion-btn button {
        border-radius: 20px !important;
        font-size: 0.82rem !important;
        padding: 4px 14px !important;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "messages": [],
        "chain": None,
        "doc_ingested": False,
        "current_file": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def handle_upload(uploaded_file):
    save_path = config.UPLOAD_DIR / uploaded_file.name
    save_path.write_bytes(uploaded_file.getvalue())

    try:
        with st.spinner("Analyzing your document..."):
            collection = sanitize_collection_name(uploaded_file.name)
            ingest_file(save_path, collection=collection)
            st.session_state.chain = build_chain(collection=collection)
            st.session_state.doc_ingested = True
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
    except Exception as e:
        st.error(f"Something went wrong: {e}")


def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 17:
        return "Good afternoon"
    return "Good evening"


def display_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.get("page", "?")
                        st.caption(f"Source {i} — Page {page}")
                        st.markdown(
                            f"<div style='background:#f7f7f8; padding:10px; "
                            f"border-radius:8px; font-size:0.85rem; "
                            f"border-left:3px solid #6366f1;'>"
                            f"{src['text'][:300]}...</div>",
                            unsafe_allow_html=True,
                        )


def main():
    init_session_state()

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📄 DocChat AI")
        st.caption("Your personal document assistant")

        st.markdown("")
        uploaded = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "docx", "md"],
            help="Supported formats: PDF, TXT, DOCX, Markdown",
        )

        if uploaded and uploaded.name != st.session_state.current_file:
            handle_upload(uploaded)

        st.divider()

        # Status section
        if st.session_state.doc_ingested:
            st.markdown(
                '<span class="status-pill status-ready">Ready</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Document:** {st.session_state.current_file}")
        else:
            st.markdown(
                '<span class="status-pill status-waiting">No document</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption(f"Model: {config.LLM_MODEL}")
        st.caption(f"Retrieval depth: {config.RETRIEVER_K} chunks")

        # New chat button
        if st.session_state.doc_ingested:
            if st.button("Clear conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # ── Main area — Welcome screen ───────────────────────────
    if not st.session_state.doc_ingested:
        st.markdown(
            f"""
            <div class="welcome-card">
                <h1>{get_greeting()}! 👋</h1>
                <p>
                    Upload a document in the sidebar and I'll help you
                    find answers, summarize sections, and clear any doubts
                    you have about it.
                </p>
                <p style="margin-top:24px; font-size:0.85rem; color:#aaa;">
                    Supports PDF, TXT, DOCX, and Markdown files
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # ── Main area — Chat ─────────────────────────────────────
    # Friendly header instead of document title
    if not st.session_state.messages:
        st.markdown(
            f"""
            <div class="welcome-card" style="padding:30px 20px;">
                <h1>Ready to help! 🎯</h1>
                <p>
                    I've read through your document. Ask me anything about it
                    — summaries, specific details, or any doubts you have.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Suggested starter questions
        st.markdown("")
        cols = st.columns(3)
        suggestions = [
            "Summarize this document",
            "What are the key points?",
            "List the important terms",
        ]
        for col, suggestion in zip(cols, suggestions):
            with col:
                if st.button(suggestion, use_container_width=True, key=suggestion):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    st.rerun()

    display_chat()

    # ── Chat input ───────────────────────────────────────────
    if prompt := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = st.session_state.chain.invoke({"question": prompt})

                answer = result["answer"]
                st.markdown(answer)

                sources = []
                for doc in result.get("source_documents", []):
                    sources.append({
                        "text": doc.page_content,
                        "page": doc.metadata.get("page", "?"),
                    })

                if sources:
                    with st.expander("View sources"):
                        for i, src in enumerate(sources, 1):
                            st.caption(f"Source {i} — Page {src['page']}")
                            st.markdown(
                                f"<div style='background:#f7f7f8; padding:10px; "
                                f"border-radius:8px; font-size:0.85rem; "
                                f"border-left:3px solid #6366f1;'>"
                                f"{src['text'][:300]}...</div>",
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as e:
                error_msg = f"Something went wrong: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })


if __name__ == "__main__":
    main()


