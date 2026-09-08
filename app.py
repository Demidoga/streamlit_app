# app.py -- Week 4 Lab: GUI (Streamlit) EDA Interface

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Lab 4 | EDA Interface",
                   layout="wide", initial_sidebar_state="expanded")

st.title("Exploratory Data Analysis Interface")

st.sidebar.header("Dataset Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV File for Analysis", type="csv")

df = pd.read_csv(uploaded_file)

if df.empty or df.columns.empty:
    st.error("The uploaded CSV has no data.")
    st.stop()

if uploaded_file is not None:

    st.subheader("Dataset Preview & Metadata")
    st.write("**First 5 Rows:**")
    st.dataframe(df.head())

    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    st.write("**Column Data Types:**")
    st.dataframe(df.dtypes.astype(str).to_frame("Data Type"))

    st.write("**Missing Values per Column:**")
    st.dataframe(pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing %": (df.isna().mean() * 100).round(2),
    }))

    st.write("**Basic Numerical Statistics:**")
    numeric = df.select_dtypes("number")
    if numeric.empty:
        st.write("No numerical columns in this dataset.")
    else:
        st.dataframe(numeric.agg(["mean", "median", "min", "max"]).T)

    st.sidebar.header("Attribute Selection")
    column = st.sidebar.selectbox("Select Attribute for Visualization",
                                df.columns)
    series = df[column].dropna()
    column_type = ("Numerical"
                if pd.api.types.is_numeric_dtype(series) else "Categorical")

    st.subheader("Visualization")
    st.caption(f"Detected attribute type: **{column_type}**")

    if series.empty:
        st.warning(f"`{column}` has no non-missing values to plot.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        if column_type == "Numerical":
            sns.histplot(series, kde=True, ax=ax)
            ax.set(title=f"Histogram of {column}", xlabel=column,
                ylabel="Frequency")
        else:
            counts = series.value_counts().head(20)  # ponytail: top 20 keeps
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
            ax.set(title=f"Frequency of {column}", xlabel=column,
                ylabel="Count")
            ax.tick_params(axis="x", rotation=45)
            for bar, n in zip(ax.patches, counts.values):
                ax.annotate(f"{n / len(series) * 100:.1f}%",
                            (bar.get_x() + bar.get_width() / 2, n),
                            ha="center", va="bottom", fontsize=8)
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to start EDA.")