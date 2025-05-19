import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import io
import time
import base64
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
start_time = time.time()


st.set_page_config(layout="wide", page_title="InsightX", page_icon="📊")
st.title("📊 InsightX")

''' 
Sidebar: File uploader and column selection
'''


st.sidebar.header("Upload Dataset")
with st.sidebar.expander("📌 About InsightX"):
    st.markdown("""
    **InsightX** is an interactive data analysis dashboard designed to empower users with quick insights from their datasets — without writing a single line of code.

    Upload your CSV or Excel files, explore data distributions, correlations, and trends visually, and quickly identify missing values or anomalies. Whether you're a data enthusiast or a professional analyst, InsightX provides a powerful yet simple interface to make data-driven decisions faster.
    """)
file = st.sidebar.file_uploader("Choose a file (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"])
if not file:
    st.info("Please upload a dataset to start analysis")
    st.stop()

if file:
    if file.name.endswith(".csv"):
        df1 = pd.read_csv(file)
    else:
        df1 = pd.read_excel(file)

    st.sidebar.header("Select Columns for Analysis")
    columns = st.sidebar.multiselect("Choose columns", df1.columns.tolist())
    df = df1[columns]


  
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2) 

    # Section 1
    st.success(f"Analysis Successfully Done in {elapsed_time} seconds")
    st.header("Quick Review")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Missing Values", df1.isnull().sum().sum())

    with col2:
        st.metric("Total Duplicate Rows", df1.duplicated().sum())

    with col3:
        dtypes_list = df1.dtypes.astype(str).unique().tolist()
        st.write("**Data Types Present**")
        st.write(dtypes_list)

    # Section 2
    st.header("Quick Analysis")
    st.subheader("Top 5 Rows")
    st.dataframe(df1.head())

    left, right = st.columns(2)

    with left:
        st.subheader("Describe")
        st.dataframe(df1.describe(include='all'))

    with right:
        st.subheader("Info")
        buffer = io.StringIO()
        df1.info(buf=buffer)
        s = "\n".join(buffer.getvalue().split("\n")[1:]) # Skip the first line
        st.text(s)


    # Encode full df1 if no columns selected, else use selected df
    target_df = df if columns else df1
    encoded_df = target_df.copy()
    for col in encoded_df.select_dtypes(include='object').columns:
        if encoded_df[col].nunique() < 10:
            encoded_df[col] = encoded_df[col].astype('category').cat.codes
        else:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
    
    # Section 3
    st.header("Visualization")
    viz1, viz2 = st.columns(2)
    viz3, viz4 = st.columns(2)

    with viz1:
        st.subheader("Feature Distributions (Histogram)")
        
        # Automatically fallback to all numerical columns if no columns are selected
        hist_cols = encoded_df[columns].select_dtypes(include='number').columns if columns else encoded_df.select_dtypes(include='number').columns
    
        num_plots = len(hist_cols)
        cols = 3  # Number of columns per row in subplot grid
        rows = -(-num_plots // cols)  # Ceiling division
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        for i, col in enumerate(hist_cols):
            axes[i].hist(encoded_df[col].dropna(), bins=20, edgecolor='black', color='skyblue')
            axes[i].set_title(col)
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        fig.suptitle("Feature Distributions", fontsize=16)
        st.pyplot(fig)


    with viz2:
        st.subheader("Box Plot")
    
        # Automatically fallback to all numerical columns if no columns are selected
        box_cols = encoded_df[columns].select_dtypes(include='number').columns if columns else encoded_df.select_dtypes(include='number').columns
    
        fig, ax = plt.subplots(figsize=(10, 5))
        encoded_df[box_cols].plot(kind='box', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    


    with viz3:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(encoded_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    with viz4:
        st.subheader("Scatter Plot")
        num_cols = encoded_df.select_dtypes(include=np.number).columns
        if len(num_cols) >= 2:
            fig = px.scatter(encoded_df, x=num_cols[0], y=num_cols[1], color=num_cols[0])
            st.plotly_chart(fig)

    st.subheader("Line Chart")
    if len(num_cols) > 0:
        fig = px.line(encoded_df[num_cols])
        st.plotly_chart(fig)

    st.subheader("Missing vs Duplicate Pie Chart")
    pie_data = pd.Series({"Missing Values": df1.isnull().sum().sum(), "Duplicate Rows": df1.duplicated().sum()})
    fig = px.pie(values=pie_data.values, names=pie_data.index, title="Missing vs Duplicates")
    st.plotly_chart(fig)

    # Export Report
    st.header("Generate Report")
    if st.button("Generate Report"):
        with st.expander("Details", expanded=True):
            st.write("This feature is under development. This will create an Ai genrated report of data that includes statistics, plots, and a summary of the analysis.")
            # Placeholder for report generation logic
            # report = generate_report(df1)
            # st.download_button("Download Report", data=report, file_name="report.pdf", mime="application/pdf")
#         report_type = st.radio("Select report format", ["PDF", "DOCX"])
#         st.info("Report generation feature to be implemented. Export includes stats, plots, and summary.")
# else:
#     st.info("Please upload a dataset to begin.")