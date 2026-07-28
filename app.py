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

st.title("🤖 AI-Powered Data Analyst Agent")
st.write("Upload your dataset (CSV, XLSX, XLS) or use the default URL to automatically perform EDA, visualize Univariate, Bivariate, and Multivariate analysis, and chat with your data!")

# Step 4: Model creation
GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

if not GOOGLE_API_KEY or not GROQ_API_KEY:
    st.warning("Please provide both Google and Groq API keys in the sidebar to proceed.")
    # Fallback or dummy setup to prevent immediate crashing if empty
    GOOGLE_API_KEY = "placeholder"
    GROQ_API_KEY = "placeholder"

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

groq_llm = ChatGroq(
    model="qwen-2.5-coder-32b-instruct",
    api_key=GROQ_API_KEY
)

# step 5: Agent Creation
def temp_tool():
  """This is just a dummy tool"""
  return "Hello world"

try:
    agent = create_agent(
        model=gemini_llm,
        tools=[temp_tool]
    )
except Exception as e:
    agent = None

def load_dataset(path: str, agent = agent):
  prompt = """return python code to read file in pandas using uploaded file extension, assume file path variable is file_path and function name is read_uploaded_file(file_path) which returns df"""

  try:
    import os
    if agent and ('file_loader.py' not in os.getcwd() or os.path.getsize('file_loader.py') == 0):
      response = agent.invoke({'messages':[{'role':'user','content':prompt}]})
      ans = response["messages"][-1].content[-1]['text']
      if "```" in ans:
          code = ans.split("```")[1]
          if code.startswith("python"):
              code = code[6:]
          with open('file_loader.py', 'w') as f:
            f.write(code)
  except Exception as e:
    # Fallback default file loader if agent fails
    with open('file_loader.py', 'w') as f:
        f.write("import pandas as pd\ndef read_uploaded_file(path):\n    if path.endswith('.csv'):\n        return pd.read_csv(path, encoding='latin1')\n    else:\n        return pd.read_excel(path)")

  return "Success"

# File uploader section in Streamlit
upload_option = st.sidebar.radio("Choose Data Source", ["Default Superstore Dataset", "Upload your own file"])

if upload_option == "Default Superstore Dataset":
    url = 'https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv'
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        if not os.path.exists("temp_data"):
            os.makedirs("temp_data")
        url = os.path.join("temp_data", uploaded_file.name)
        with open(url, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        url = 'https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv'

load_dataset(url, agent)

def read_file(path):
  try:
      from file_loader import read_uploaded_file
      return read_uploaded_file(path)
  except Exception as e:
      if path.endswith('.csv'):
          return pd.read_csv(path, encoding='latin1')
      return pd.read_excel(path)

try:
    df = read_file(url)
    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    df = pd.DataFrame()

# tool 3
def perform_eda_func(data, agent):
  if data.empty:
      return "Empty Data"
      
  df_sample = data.sample(min(5, len(data)))
  prompt = f"""You are a data analysts perform basic eda python single function perform_eda code and give all required analysis like missing values and columns. Data frame sample : {df_sample} data stats: {df_sample.describe()}"""

  try:
      if agent:
          response = agent.invoke({'messages':[{'role':'user','content':prompt}]})
          ans = response["messages"][-1].content[-1]['text']
          code = ans.split("```")[1]
          if code.startswith("python"):
              code = code[6:]
      else:
          raise Exception()
  except:
      code = """def perform_eda(df):
    print("--- Missing Values ---")
    print(df.isnull().sum())
    print("--- Data Info ---")
    print(df.info())
    print("--- Summary Statistics ---")
    print(df.describe())
"""

  with open('basic_eda.py', 'w') as f:
    f.write(code)


  # ============Advance EDA========
  advance_prompt = """give detailed prompt for advance data analysis, which must include describe, corr, univariate numerical and object column analysis bivariate analysis, time series if any date column given multivariate analysis to perform different col like example sales, region, segment using bar plot with hue, give code with strict python and module code with pip install for any unknown new module if required"""

  try:
      if agent:
          response = agent.invoke({'messages':[{'role':'user','content':advance_prompt}]})
          system_prompt_model = response["messages"][-1].content[-1]['text']

          new_prompt = """Give Python advance_eda.py file with every code inside a single function eda_by_ai and no need to load file, df is already loaded, starts with using df and return streamlit plots or matplotlib figures. """ + system_prompt_model
          response = agent.invoke({'messages':[{'role':'user','content':new_prompt}]})
          ans = response["messages"][-1].content[-1]['text']
          code_adv = ans.split("```")[1]
          if code_adv.startswith("python"):
              code_adv = code_adv[6:]
      else:
          raise Exception()
  except:
      code_adv = """import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def eda_by_ai(df):
    st.subheader("Univariate & Bivariate Analysis")
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[numeric_cols[0]], kde=True, ax=ax)
        st.pyplot(fig)
        
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
"""

  with open('advance_eda.py', 'w') as f:
    f.write(code_adv)

  return "Success"

if not df.empty and st.button("Run Automated EDA Analysis"):
    with st.spinner("Generating EDA code and running analysis..."):
        perform_eda_func(df, agent)
        st.success("EDA Scripts Generated!")
        
        try:
            from basic_eda import perform_eda
            st.subheader("Basic EDA Output")
            import io
            import sys
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            perform_eda(df)
            sys.stdout = old_stdout
            st.text(new_stdout.getvalue())
        except Exception as e:
            st.write("Basic EDA executed via fallback.")
            st.write(df.describe())
            
        try:
            from advance_eda import eda_by_ai
            st.subheader("Advanced EDA & Visualizations")
            eda_by_ai(df)
        except Exception as e:
            st.error(f"Error running advanced EDA: {e}")

# Chat with Data Feature
st.markdown("---")
st.subheader("💬 Chat with your Data")
user_query = st.text_input("Ask anything about your dataset (e.g., 'What is the total sales by region?'):")

if user_query and not df.empty:
    with st.spinner("Analyzing query and executing code..."):
        chat_prompt = f"""You are an expert pandas data analyst. The dataframe 'df' is already loaded in memory with columns {list(df.columns)}. 
        Write executable Python code to answer the user's question: '{user_query}'. 
        Save any generated plot to a matplotlib figure or print the text result. Return ONLY executable python code block."""
        
        try:
            if agent:
                response = agent.invoke({'messages':[{'role':'user','content':chat_prompt}]})
                ans = response["messages"][-1].content[-1]['text']
                code = ans.split("```")[1]
                if code.startswith("python"):
                    code = code[6:]
            else:
                code = "print(df.head())"
                
            st.text("Generated Code Execution:")
            st.code(code, language='python')
            
            # Execute code safely in local context
            local_vars = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns, "st": st}
            exec(code, {}, local_vars)
        except Exception as e:
            st.error(f"Error processing query: {e}")
            # Fallback simple query executor
            if "sales" in user_query.lower() and "Sales" in df.columns:
                st.write(df.groupby(df.columns[0])['Sales'].sum().reset_index())
            else:
                st.write(df.describe())