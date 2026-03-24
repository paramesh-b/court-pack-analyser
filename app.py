import streamlit as st
import os
import pandas as pd

from extractor import extract_text_from_pdf, load_sample_text
from analyser import analyse_claim
from logger import log_result

# RAG imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document

st.set_page_config(
    page_title="Court Pack Analyser",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ AI Court Pack Analyser")
st.markdown("**Automated motor insurance claim analysis powered by AI**")
st.divider()

# ── INPUT ──────────────────────────────────────────────────────────────────────
option = st.radio("Choose input method:", ["Use sample court pack", "Upload PDF"])

text = ""

if option == "Use sample court pack":
    text = load_sample_text()
    st.success("Sample court pack loaded.")
    with st.expander("View raw document text"):
        st.text(text)

elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a court pack PDF", type=["pdf"])
    if uploaded_file:
        os.makedirs("sample_docs", exist_ok=True)
        with open("sample_docs/temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        text = extract_text_from_pdf("sample_docs/temp.pdf")
        st.success("PDF uploaded and text extracted.")
        with st.expander("View raw document text"):
            st.text(text)

# store text in session so it persists after button clicks
if text:
    st.session_state["doc_text"] = text

# ── CLAIM ANALYSIS ─────────────────────────────────────────────────────────────
if st.session_state.get("doc_text") and st.button("🔍 Analyse Claim", type="primary"):
    with st.spinner("Analysing claim with AI..."):
        try:
            result = analyse_claim(st.session_state["doc_text"])
            log_result(result)
            st.session_state["result"] = result
        except Exception as e:
            st.error("⚠️ Analysis failed. The document may not contain recognisable claim data. Please try a different file.")
            st.stop()

if "result" in st.session_state:
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
    risk = result["risk_level"]

    if risk == "HIGH":
        st.error(f"⛔ Risk Level: HIGH — Rate inflated by {deviation}% above benchmark")
    elif risk == "MEDIUM":
        st.warning(f"⚠️ Risk Level: MEDIUM — Rate inflated by {deviation}% above benchmark")
    else:
        st.success(f"✅ Risk Level: LOW — Rate within {deviation}% of benchmark")

    st.info(f"💡 Recommendation: {result['recommendation']}")

# ── AUDIT LOG ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("📊 Audit Log")
if st.button("View Analysis History"):
    try:
        log_df = pd.read_csv("data/analysis_log.csv")
        st.dataframe(log_df, use_container_width=True)
        csv = log_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Audit Log",
            data=csv,
            file_name="court_pack_audit_log.csv",
            mime="text/csv"
        )
    except FileNotFoundError:
        st.info("No analyses run yet.")

# ── RAG QUERY SECTION ──────────────────────────────────────────────────────────
if st.session_state.get("doc_text"):
    st.divider()
    st.subheader("💬 Ask a Question About This Document")
    st.markdown(
        "Type any question in plain English. "
        "The AI will search the document and answer from what it actually says — "
        "not from guesswork."
    )

    @st.cache_resource
    def build_vectorstore(doc_text: str):
        """Split document into chunks, embed them, store in FAISS."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        chunks = splitter.split_text(doc_text)
        docs = [Document(page_content=chunk) for chunk in chunks]
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore

    question = st.text_input(
        "Your question:",
        placeholder="e.g. What was the daily hire rate?  Who is the claimant?  How long was the hire period?"
    )

    if question:
        with st.spinner("Searching document and generating answer..."):
            try:
                groq_key = os.getenv("GROQ_API_KEY")

                vectorstore = build_vectorstore(st.session_state["doc_text"])
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0,
                    api_key=groq_key
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )

                response = qa_chain({"query": question})

                st.success("**Answer:**")
                st.write(response["result"])

                with st.expander("📄 Source sections used to generate this answer"):
                    for i, doc in enumerate(response["source_documents"], 1):
                        st.markdown(f"**Section {i}:**")
                        st.text(doc.page_content)
                        st.markdown("---")

            except Exception as e:
                st.error(f"Could not answer question. Error: {str(e)}")
