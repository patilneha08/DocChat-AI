"""
Minimal debug version — isolates exactly where the failure happens.
Run:  streamlit run app_debug.py
"""
import streamlit as st

st.set_page_config(page_title="DocChat AI Debug", page_icon="📄", layout="wide")
st.title("📄 DocChat AI — Debug Mode")
st.write("✅ Step 1: Streamlit is rendering.")

# ── Step 2: Test config import ───────────────────────────────
try:
    import config
    st.write(f"✅ Step 2: config.py loaded. UPLOAD_DIR = `{config.UPLOAD_DIR}`")
except Exception as e:
    st.error(f"❌ Step 2 FAILED — config.py: {e}")
    st.stop()

# ── Step 3: Test ingest import ───────────────────────────────
try:
    from ingest import ingest_file, load_vectorstore, get_embeddings
    st.write("✅ Step 3: ingest.py imported.")
except Exception as e:
    st.error(f"❌ Step 3 FAILED — ingest.py: {e}")
    st.stop()

# ── Step 4: Test chain import ────────────────────────────────
try:
    from chain import build_chain, get_llm
    st.write("✅ Step 4: chain.py imported.")
except Exception as e:
    st.error(f"❌ Step 4 FAILED — chain.py: {e}")
    st.stop()

# ── Step 5: Test embedding model loads ───────────────────────
try:
    with st.spinner("Loading embedding model (first run downloads ~80MB)..."):
        emb = get_embeddings()
        test_vec = emb.embed_query("test")
    st.write(f"✅ Step 5: Embeddings working. Vector dim = {len(test_vec)}")
except Exception as e:
    st.error(f"❌ Step 5 FAILED — Embeddings: {e}")
    st.stop()

# ── Step 6: Test Ollama connection ───────────────────────────
try:
    import urllib.request
    import json
    resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    data = json.loads(resp.read())
    models = [m["name"] for m in data.get("models", [])]
    st.write(f"✅ Step 6: Ollama running. Available models: `{models}`")

    if not any(config.LLM_MODEL in m for m in models):
        st.warning(
            f"⚠️ Model `{config.LLM_MODEL}` not found! "
            f"Run: `ollama pull {config.LLM_MODEL}`"
        )
except Exception as e:
    st.error(f"❌ Step 6 FAILED — Ollama not reachable: {e}")
    st.info("Run `ollama serve` in another terminal, then refresh.")
    st.stop()

# ── Step 7: Test LLM responds ───────────────────────────────
try:
    with st.spinner("Testing LLM response..."):
        llm = get_llm()
        reply = llm.invoke("Say 'hello' in one word.")
    st.write(f"✅ Step 7: LLM responded: `{reply.content.strip()[:100]}`")
except Exception as e:
    st.error(f"❌ Step 7 FAILED — LLM invoke: {e}")
    st.stop()

# ── Step 8: Test file upload + full pipeline ─────────────────
st.divider()
st.subheader("🧪 Full Pipeline Test")

uploaded = st.file_uploader("Upload a test document", type=["pdf", "txt", "docx", "md"])

if uploaded:
    try:
        save_path = config.UPLOAD_DIR / uploaded.name
        save_path.write_bytes(uploaded.getvalue())
        st.write(f"✅ File saved to `{save_path}`")

        with st.spinner("Ingesting..."):
            from config import sanitize_collection_name
            collection = sanitize_collection_name(uploaded.name)
            vs = ingest_file(save_path, collection=collection)
        st.write(f"✅ Ingested into collection `{collection}`")

        with st.spinner("Building chain..."):
            chain = build_chain(collection=collection)
        st.write("✅ Chain built successfully")

        question = st.text_input("Ask a test question:")
        if question:
            with st.spinner("Querying..."):
                result = chain.invoke({"question": question})
            st.write("**Answer:**", result["answer"])
            st.write(f"**Sources:** {len(result.get('source_documents', []))} chunks")

    except Exception as e:
        st.error(f"❌ Pipeline failed: {e}")
        import traceback
        st.code(traceback.format_exc())

st.divider()
st.success("🎉 All basic checks passed! The full app.py should work fine.")