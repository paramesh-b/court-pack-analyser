import streamlit as st
import os
import pandas as pd

from extractor import extract_text_from_pdf, load_sample_text
from analyser import analyse_claim, validate_document
from logger import log_result
from rag_pipeline import CourtPackRAG

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Court Pack Analyser",
    page_icon="⚖️",
    layout="wide"
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 Document Library")
    st.markdown("Previously indexed documents (local only).")

    indexed_docs = CourtPackRAG.list_indexed_documents()

    if not indexed_docs:
        st.info("No persisted documents. Upload documents to index them.")
    else:
        st.success(f"{len(indexed_docs)} document(s) in store")
        for doc in indexed_docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                label = f"📄 {doc['filename']} ({doc['chunk_count']} chunks)"
                if st.button(label, key=f"load_{doc['collection_name']}"):
                    with st.spinner("Loading from ChromaDB..."):
                        rag = CourtPackRAG()
                        n = rag.load_existing(doc["collection_name"])
                        st.session_state["rag"]              = rag
                        st.session_state["rag_text"]         = doc["collection_name"]
                        st.session_state["active_doc_label"] = doc["filename"]
                        st.session_state["doc_text"]         = None
                        st.session_state.pop("chat_history", None)
                    st.success(f"Loaded — {n} chunks ready")
            with col2:
                if st.button("🗑️", key=f"del_{doc['collection_name']}"):
                    CourtPackRAG.delete_document(doc["collection_name"])
                    st.rerun()

    st.divider()
    st.caption("Vector store: ChromaDB (local) / FAISS (cloud)")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("LLM: Groq LLaMA 3.1-8b-instant")

# ── MAIN ───────────────────────────────────────────────────────────────────────
st.title("⚖️ AI Court Pack Analyser")
st.markdown("**Automated motor insurance claim analysis powered by RAG**")
st.divider()

mode = st.radio(
    "Choose mode:",
    ["Single document", "Batch processing", "Use sample court pack"],
    horizontal=True
)

# ── SINGLE DOCUMENT ────────────────────────────────────────────────────────────
if mode == "Single document":
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file:
        os.makedirs("sample_docs", exist_ok=True)
        with open("sample_docs/temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        text = extract_text_from_pdf("sample_docs/temp.pdf")
        st.success(f"Uploaded: {uploaded_file.name}")
        st.session_state["doc_text"]         = text
        st.session_state["active_doc_label"] = uploaded_file.name
        st.session_state.pop("batch_docs", None)
        with st.expander("View raw document text"):
            st.text(text)

# ── SAMPLE ─────────────────────────────────────────────────────────────────────
elif mode == "Use sample court pack":
    text = load_sample_text()
    st.success("Sample court pack loaded.")
    st.session_state["doc_text"]         = text
    st.session_state["active_doc_label"] = "sample_court_pack"
    st.session_state.pop("batch_docs", None)
    with st.expander("View raw document text"):
        st.text(text)

# ── BATCH PROCESSING ───────────────────────────────────────────────────────────
elif mode == "Batch processing":
    st.markdown(
        "Upload multiple PDFs. Court pack documents will be analysed automatically. "
        "All documents are indexed for Q&A."
    )
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Process All Documents", type="primary"):
        os.makedirs("sample_docs", exist_ok=True)

        batch_results = []
        indexed_names = []
        skipped_names = []
        batch_docs    = {}  # filename → text (for Q&A switching)

        progress = st.progress(0, text="Starting...")

        for i, file in enumerate(uploaded_files):
            progress.progress(
                int((i / len(uploaded_files)) * 100),
                text=f"Processing {file.name}..."
            )

            temp_path = f"sample_docs/batch_temp_{i}.pdf"
            with open(temp_path, "wb") as f:
                f.write(file.read())

            try:
                text = extract_text_from_pdf(temp_path)
            except Exception as e:
                st.warning(f"⚠️ Could not extract {file.name}: {e}")
                try: os.remove(temp_path)
                except: pass
                continue

            # Store text for Q&A
            batch_docs[file.name] = text

            # Index into vector store
            try:
                rag      = CourtPackRAG()
                n_chunks = rag.index_document(text, filename=file.name)
                indexed_names.append(f"{file.name} ({n_chunks} chunks)")
            except Exception as e:
                st.warning(f"⚠️ Could not index {file.name}: {e}")
                try: os.remove(temp_path)
                except: pass
                continue

            # Validate for claim analysis
            is_valid, _ = validate_document(text)
            if not is_valid:
                skipped_names.append(file.name)
                try: os.remove(temp_path)
                except: pass
                continue

            # Run claim analysis
            try:
                result = analyse_claim(text)
                log_result(result)
                result["filename"] = file.name
                batch_results.append(result)
            except Exception as e:
                st.warning(f"⚠️ Analysis failed for {file.name}: {e}")

            try: os.remove(temp_path)
            except: pass

        progress.progress(100, text="✅ Complete.")

        # Save everything to session state
        st.session_state["batch_results"] = batch_results
        st.session_state["batch_indexed"] = indexed_names
        st.session_state["batch_skipped"] = skipped_names
        st.session_state["batch_total"]   = len(uploaded_files)
        st.session_state["batch_done"]    = True
        st.session_state["batch_docs"]    = batch_docs  # all texts for Q&A
        st.session_state.pop("doc_text", None)
        st.session_state.pop("rag", None)
        st.session_state.pop("rag_text", None)
        st.session_state.pop("chat_history", None)

# ── BATCH RESULTS ──────────────────────────────────────────────────────────────
if mode == "Batch processing" and st.session_state.get("batch_done"):
    batch_results = st.session_state.get("batch_results", [])
    indexed_names = st.session_state.get("batch_indexed", [])
    skipped_names = st.session_state.get("batch_skipped", [])
    total         = st.session_state.get("batch_total", 0)

    st.divider()
    st.subheader("📦 Batch Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Uploaded", total)
    col2.metric("Court Packs Analysed", len(batch_results))
    col3.metric("Non-Court-Pack (Q&A only)", len(skipped_names))

    if indexed_names:
        with st.expander(f"✅ {len(indexed_names)} document(s) indexed"):
            for name in indexed_names:
                st.write(f"• {name}")

    if skipped_names:
        with st.expander(f"ℹ️ {len(skipped_names)} skipped for claim analysis"):
            for name in skipped_names:
                st.write(f"• {name}")

    if batch_results:
        st.divider()
        st.subheader("📋 Claim Analysis Results")

        df = pd.DataFrame(batch_results)[[
            "filename", "claimant", "hire_company", "vehicle_category",
            "region", "hire_duration_days", "daily_rate_charged",
            "benchmark_daily_rate", "rate_deviation_pct", "risk_level",
            "total_claim", "recommendation"
        ]]

        def highlight_risk(val):
            if val == "HIGH":     return "background-color: #ffcccc"
            elif val == "MEDIUM": return "background-color: #fff3cc"
            return "background-color: #ccffcc"

        st.dataframe(
            df.style.applymap(highlight_risk, subset=["risk_level"]),
            use_container_width=True
        )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Batch Results CSV",
            data=csv,
            file_name="batch_claim_analysis.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("📊 Risk Distribution")
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        st.bar_chart(risk_counts.set_index("Risk Level"))
    else:
        st.info("No valid court pack documents found in the batch.")

# ── SINGLE DOC ANALYSE BUTTON ──────────────────────────────────────────────────
if mode in ["Single document", "Use sample court pack"] and st.session_state.get("doc_text"):
    if st.button("🔍 Analyse Claim", type="primary"):
        is_valid, reason = validate_document(st.session_state["doc_text"])
        if not is_valid:
            st.error("❌ This does not appear to be a motor insurance court pack.")
            st.warning("💡 You can still use Q&A below.")
            st.session_state.pop("result", None)
        else:
            with st.spinner("Analysing..."):
                try:
                    result = analyse_claim(st.session_state["doc_text"])
                    log_result(result)
                    st.session_state["result"] = result
                    st.session_state.pop("rag", None)
                    st.session_state.pop("rag_text", None)
                    st.session_state.pop("chat_history", None)
                except Exception:
                    st.error("⚠️ Analysis failed.")

if "result" in st.session_state and mode in ["Single document", "Use sample court pack"]:
    result = st.session_state["result"]
    st.divider()
    st.subheader("📋 Claim Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Claimant", result["claimant"])
        st.metric("Hire Company", result["hire_company"])
        st.metric("Vehicle Category", result["vehicle_category"])
        st.metric("Region", result["region"])
    with col2:
        st.metric("Hire Duration", f"{result['hire_duration_days']} days")
        st.metric("Daily Rate Charged", f"£{result['daily_rate_charged']}")
        st.metric("Benchmark Rate", f"£{result['benchmark_daily_rate']}")
        st.metric("Total Claim", f"£{result['total_claim']}")

    st.divider()
    st.subheader("🚨 Risk Assessment")
    deviation = result["rate_deviation_pct"]
    risk      = result["risk_level"]
    if risk == "HIGH":
        st.error(f"⛔ Risk Level: HIGH — Rate inflated by {deviation}% above benchmark")
    elif risk == "MEDIUM":
        st.warning(f"⚠️ Risk Level: MEDIUM — Rate inflated by {deviation}% above benchmark")
    else:
        st.success(f"✅ Risk Level: LOW — Rate within {deviation}% of benchmark")
    st.info(f"💡 Recommendation: {result['recommendation']}")

# ── AUDIT LOG ──────────────────────────────────────────────────────────────────
if st.session_state.get("doc_text") and mode in ["Single document", "Use sample court pack"]:
    st.divider()
    st.subheader("📊 Audit Log")
    if st.button("View Analysis History"):
        try:
            log_df = pd.read_csv("data/analysis_log.csv")
            st.dataframe(log_df, use_container_width=True)
            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Audit Log", csv, "court_pack_audit_log.csv", "text/csv")
        except FileNotFoundError:
            st.info("No analyses run yet.")

# ── RAG Q&A ────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("💬 Ask Questions About a Document")

# Determine Q&A source
batch_docs = st.session_state.get("batch_docs", {})
doc_text   = st.session_state.get("doc_text")
rag_ready  = "rag" in st.session_state

if not doc_text and not rag_ready and not batch_docs:
    st.info("👈 Upload a document or select one from the sidebar to start asking questions.")
else:
    # ── BATCH MODE: show document selector dropdown ────────────────────────────
    if mode == "Batch processing" and batch_docs:
        selected_doc = st.selectbox(
            "Select a document to ask questions about:",
            options=list(batch_docs.keys()),
            key="batch_doc_selector"
        )

        # Load selected doc into RAG if changed
        selected_text = batch_docs[selected_doc]
        if (
            "rag" not in st.session_state or
            st.session_state.get("active_doc_label") != selected_doc
        ):
            with st.spinner(f"Loading {selected_doc} for Q&A..."):
                rag = CourtPackRAG()
                n   = rag.index_document(selected_text, filename=selected_doc)
                st.session_state["rag"]              = rag
                st.session_state["rag_text"]         = selected_doc
                st.session_state["active_doc_label"] = selected_doc
                st.session_state.pop("chat_history", None)
            st.caption(f"✅ {n} chunks embedded with all-MiniLM-L6-v2")

    # ── SINGLE/SAMPLE MODE: build RAG from doc_text ────────────────────────────
    elif doc_text and (
        "rag" not in st.session_state or
        st.session_state.get("rag_text") != doc_text
    ):
        with st.spinner("Indexing into vector store..."):
            rag   = CourtPackRAG()
            label = st.session_state.get("active_doc_label", "Unknown")
            n     = rag.index_document(doc_text, filename=label)
            st.session_state["rag"]      = rag
            st.session_state["rag_text"] = doc_text
        st.caption(f"✅ {n} chunks embedded with all-MiniLM-L6-v2")

    # ── CHAT INTERFACE ─────────────────────────────────────────────────────────
    if "rag" in st.session_state:
        active_label = st.session_state.get("active_doc_label", "Current document")
        st.markdown(f"**Active document:** `{active_label}`")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_question = st.chat_input(
            "e.g. What is this document about?  Who is the claimant?  What was the daily hire rate?"
        )

        if user_question:
            with st.chat_message("user"):
                st.markdown(user_question)
            st.session_state["chat_history"].append({"role": "user", "content": user_question})

            with st.spinner("Searching vector store..."):
                context = st.session_state["rag"].retrieve(user_question, k=3)

            groq_key = os.getenv("GROQ_API_KEY")
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=groq_key)

            prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are an AI document analyst powered by a RAG (Retrieval-Augmented Generation) pipeline. "
                 "When asked about yourself, explain that you use RAG: documents are chunked, embedded with "
                 "all-MiniLM-L6-v2, stored in a vector store, and the most relevant chunks are "
                 "retrieved via semantic similarity search to answer each question. "
                 "Answer using ONLY the retrieved document sections provided below. "
                 "If the answer is not present, say exactly: "
                 "'This information is not found in the document.' "
                 "Do not guess or use outside knowledge.\n\nDocument sections:\n{context}"),
                ("human", "{question}")
            ])

            with st.spinner("Generating answer..."):
                chain    = prompt | llm
                response = chain.invoke({"context": context, "question": user_question})
                answer   = response.content.strip()

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

            with st.expander("📄 Retrieved chunks from vector store"):
                st.text(context)

        if st.session_state.get("chat_history"):
            if st.button("🗑️ Clear chat history"):
                st.session_state["chat_history"] = []
                st.rerun()