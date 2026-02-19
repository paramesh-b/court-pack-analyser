import streamlit as st
from extractor import extract_text_from_pdf, load_sample_text
from analyser import analyse_claim
from logger import log_result
import pandas as pd

st.set_page_config(
    page_title="Court Pack Analyser",
    page_icon="⚖️",
    layout="centered"
)

st.title("⚖️ AI Court Pack Analyser")
st.markdown("**Automated motor insurance claim analysis powered by AI**")
st.divider()

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
        with open("sample_docs/temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        text = extract_text_from_pdf("sample_docs/temp.pdf")
        st.success("PDF uploaded and text extracted.")
        with st.expander("View raw document text"):
            st.text(text)

if text and st.button("🔍 Analyse Claim", type="primary"):
    with st.spinner("Analysing claim with AI..."):
        result = analyse_claim(text)
        log_result(result)

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