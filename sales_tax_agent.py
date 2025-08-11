import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Set your OpenAI API key securely
os.environ["OPENAI_API_KEY"] = "sk-...your-key..."

st.set_page_config(page_title="Sales Tax Analyzer", layout="wide")
st.title("🧾 Sales Tax Data Analyzer with LLM + Pandas Agent")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your sales tax CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    st.success("CSV loaded successfully!")
    st.write("Preview of data:")
    st.dataframe(df.head())

    # Optional: Anomaly detection
    if st.checkbox("Run basic anomaly detection on collected_tax"):
        if 'collected_tax' in df.columns:
            df['tax_zscore'] = zscore(df['collected_tax'].fillna(0))
            anomalies = df[df['tax_zscore'].abs() > 2]
            st.markdown("### 🚨 Detected Anomalies in 'collected_tax'")
            st.dataframe(anomalies)
        else:
            st.warning("Column 'collected_tax' not found in your data.")

    # Optional: Chart rendering
    if st.checkbox("Show collected tax by state chart"):
        if 'state' in df.columns and 'collected_tax' in df.columns:
            fig, ax = plt.subplots()
            sns.barplot(x='state', y='collected_tax', data=df, estimator=sum, ci=None, ax=ax)
            ax.set_title("Total Collected Tax by State")
            st.pyplot(fig)
        else:
            st.warning("Required columns 'state' and 'collected_tax' not found.")

    # Initialize the LLM and agent
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        agent_type="openai-tools",
        allow_dangerous_code=True
    )

    st.markdown("### Ask a question about your data:")
    user_input = st.text_area("Your question", placeholder="e.g., Which state had the highest local tax rate in Q1 2024?")

    if st.button("Submit") and user_input:
        with st.spinner("Analyzing with agent..."):
            try:
                response = agent.run(user_input)
                st.markdown("### 🧠 Assistant Response")
                st.write(response)

                # Optional: Export response as text
                st.download_button(
                    label="📄 Download Response as TXT",
                    data=io.StringIO(response),
                    file_name="agent_response.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a CSV file to get started.")
