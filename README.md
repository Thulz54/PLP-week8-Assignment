import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("Python Frameworks Assignment: CORD-19 Metadata Analysis")
st.markdown("""
This app loads the metadata.csv file from the CORD-19 dataset, 
performs basic analysis, and visualizes some trends.
""")

st.header("1. Load Data")
try:
    df = pd.read_csv("metadata.csv", parse_dates=["publish_time"], dayfirst=True)
    st.success("metadata.csv loaded successfully!")
except FileNotFoundError:
    st.error("metadata.csv not found. Please place it in the same folder as this app.")
    st.stop()

    st.header("2. Data Exploration")
st.subheader("First 5 rows:")
st.dataframe(df.head())

st.subheader("DataFrame shape:")
st.write(f"{df.shape[0]} rows and {df.shape[1]} columns")

st.subheader("Missing values per column:")
st.write(df.isnull().sum())

st.subheader("Data types of each column:")
st.write(df.dtypes)
st.header("3. Data Cleaning")

categorical_cols = df.select_dtypes(include="object").columns
df[categorical_cols] = df[categorical_cols].fillna("Unknown")
# Convert publish_time to datetime

st.write("Filled missing values in categorical columns with 'Unknown'.")
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
st.write("Converted publish_time to datetime format.")
df["publish_time"] = df["publish_time"].fillna(pd.Timestamp("1970-01-01"))
st.write("Filled missing publish_time with '1970-01-01'.")

st.header("4. Numerical Column Statistics")
st.write(df.describe())
numerical_cols = df.select_dtypes(include=["number"]).columns
st.subheader("Numerical Columns Statistics:")

st.header("5. Visualizations")
st.subheader("Publications Over Time")
df["year_month"] = df["publish_time"].dt.to_period("M")

st.subheader("Number of papers per journal (top 10)")
top_journals = df['journal'].value_counts().head(10)
fig1, ax1 = plt.subplots()
sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax1, palette="viridis")
ax1.set_xlabel("Number of Papers")
ax1.set_ylabel("Journal")
st.pyplot(fig1)

st.subheader("Paper distribution by year")
df['year'] = df['publish_time'].dt.year
fig2, ax2 = plt.subplots()
df['year'].value_counts().sort_index().plot(kind='bar', ax=ax2, color='skyblue')
ax2.set_xlabel("Year")
ax2.set_ylabel("Number of Papers")
st.pyplot(fig2)

st.subheader("Top 10 authors (based on occurrence)")
authors_series = df['authors'].str.split(';').explode()
top_authors = authors_series.value_counts().head(10)
fig3, ax3 = plt.subplots()
sns.barplot(x=top_authors.values, y=top_authors.index, ax=ax3, palette="magma")
ax3.set_xlabel("Number of Papers")
ax3.set_ylabel("Author")
st.pyplot(fig3)

st.success("Analysis complete!")
