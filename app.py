import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.agents import create_agent

st.set_page_config(page_title="AI Powered Data Analyst Agent", layout="wide")
st.title("AI Powered Data Analyst Agent")

GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.warning("Please enter both Google and Groq API keys in the sidebar to proceed.")
    st.stop()

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

groq_llm = ChatGroq(
    model="qwen-2.5-coder-32b-instruct",
    api_key=GROQ_API_KEY
)

def temp_tool():
    return "Hello world"

agent = create_agent(
    model=gemini_llm,
    tools=[temp_tool]
)

uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
url_input = st.sidebar.text_input("Or enter Dataset URL (CSV/Excel)")

data_source = None
if uploaded_file is not None:
    data_source = uploaded_file
elif url_input:
    data_source = url_input

if data_source is not None:
    try:
        if isinstance(data_source, str) and data_source.endswith(('xlsx', 'xls')):
            df = pd.read_excel(data_source)
        elif isinstance(data_source, str):
            df = pd.read_csv(data_source, encoding='latin1')
        else:
            df = pd.read_csv(uploaded_file, encoding='latin1')
        
        st.success("Dataset Loaded Successfully!")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Basic & Advanced EDA", 
        "Univariate Analysis", 
        "Bivariate Analysis", 
        "Multivariate Analysis", 
        "Chat with Data Features"
    ])

    with tab1:
        st.subheader("Automatic EDA Generation & Execution")
        if st.button("Run Full EDA Analysis"):
            with st.spinner("Agent is generating and executing analysis..."):
                try:
                    df_sample = df.sample(min(5, len(df)))
                    prompt = f"""You are a data analyst. Write a python function named perform_eda(df) that prints basic eda like shape, missing values, and columns using dataframe sample :{df_sample} and stats: {df_sample.describe()}. Return only executable python code in code blocks."""
                    
                    response = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
                    ans = response["messages"][-1].content
                    if isinstance(ans, list):
                        ans = ans[-1].get('text', str(ans))
                    code = ans.split("```")[1]
                    if code.startswith("python"):
                        code = code[6:]
                    with open('basic_eda.py', 'w') as f:
                        f.write(code)

                    advance_prompt = """Give Python advance_eda.py file with every code inside a single function eda_by_ai(df) and no need to load file, df is already loaded. Perform describe, corr, univariate numerical and object column analysis, bivariate analysis, and multivariate analysis. Use matplotlib and seaborn. Return only code."""
                    
                    response2 = agent.invoke({'messages': [{'role': 'user', 'content': advance_prompt}]})
                    ans2 = response2["messages"][-1].content
                    if isinstance(ans2, list):
                        ans2 = ans2[-1].get('text', str(ans2))
                    code2 = ans2.split("```")[1]
                    if code2.startswith("python"):
                        code2 = code2[6:]
                    with open('advance_eda.py', 'w') as f:
                        f.write(code2)

                    from basic_eda import perform_eda
                    from advance_eda import eda_by_ai

                    st.text("--- Basic EDA Output ---")
                    perform_eda(df)
                    st.text("--- Advanced EDA Output ---")
                    eda_by_ai(df)
                    st.pyplot(plt.gcf())
                except Exception as e:
                    st.error(f"Error executing EDA: {e}")

    with tab2:
        st.subheader("Univariate Analysis")
        uni_col = st.selectbox("Select Column for Univariate Analysis", df.columns, key="uni")
        if st.button("Generate Univariate Plot"):
            fig, ax = plt.subplots(figsize=(8, 5))
            if pd.api.types.is_numeric_dtype(df[uni_col]):
                sns.histplot(df[uni_col], kde=True, ax=ax, color='blue')
                ax.set_title(f'Distribution of {uni_col}')
            else:
                top_cats = df[uni_col].value_counts().head(10)
                sns.barplot(x=top_cats.index, y=top_cats.values, ax=ax, palette='viridis')
                ax.set_title(f'Frequency of {uni_col}')
                plt.xticks(rotation=45)
            st.pyplot(fig)

    with tab3:
        st.subheader("Bivariate Analysis")
        col1 = st.selectbox("Select X-axis Column", df.columns, key="bi_x")
        col2 = st.selectbox("Select Y-axis Column", df.columns, key="bi_y")
        if st.button("Generate Bivariate Plot"):
            fig, ax = plt.subplots(figsize=(8, 5))
            if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                sns.scatterplot(data=df, x=col1, y=col2, ax=ax, color='purple')
                ax.set_title(f'{col1} vs {col2}')
            elif not pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
                sns.boxplot(data=df, x=col1, y=col2, ax=ax, palette='Set2')
                ax.set_title(f'{col2} by {col1}')
                plt.xticks(rotation=45)
            else:
                sns.countplot(data=df, x=col1, hue=col2, ax=ax)
                ax.set_title(f'{col1} grouped by {col2}')
                plt.xticks(rotation=45)
            st.pyplot(fig)

    with tab4:
        st.subheader("Multivariate Analysis")
        m_col1 = st.selectbox("Select X Column", df.columns, key="multi_x")
        m_col2 = st.selectbox("Select Y Column", df.columns, key="multi_y")
        m_hue = st.selectbox("Select Hue Column", df.columns, key="multi_hue")
        if st.button("Generate Multivariate Plot"):
            fig, ax = plt.subplots(figsize=(10, 6))
            if pd.api.types.is_numeric_dtype(df[m_col1]) and pd.api.types.is_numeric_dtype(df[m_col2]):
                sns.scatterplot(data=df, x=m_col1, y=m_col2, hue=m_hue, ax=ax, palette='deep')
            else:
                sns.barplot(data=df, x=m_col1, y=m_col2, hue=m_hue, ax=ax, palette='muted')
            ax.set_title(f'Multivariate Analysis: {m_col1} vs {m_col2} with Hue {m_hue}')
            plt.xticks(rotation=45)
            st.pyplot(fig)

    with tab5:
        st.subheader("Chat with Data Features")
        user_query = st.text_input("Ask anything about your dataset features or request custom analysis/code:")
        if user_query:
            with st.spinner("AI is thinking..."):
                chat_prompt = f"""Dataset columns and types: {df.dtypes.to_dict()}. Dataset sample: {df.head(2).to_dict()}. User query: {user_query}. Provide a helpful and clear response, and python code if requested."""
                response = agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]})
                ans = response["messages"][-1].content
                if isinstance(ans, list):
                    ans = ans[-1].get('text', str(ans))
                st.write(ans)
else:
    st.info("Please upload a dataset or provide a valid URL from the sidebar to begin.")