import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("student_table_enriched.csv")
    return df

df = load_data()

# Identify RIT columns
rit_cols = [col for col in df.columns if col.startswith("rit_")]

# Sidebar filters
st.sidebar.title("Filters")
selected_grade = st.sidebar.multiselect("Grade (2015)", sorted(df["grade_2015"].dropna().unique()))
selected_school = st.sidebar.multiselect("School (2015)", sorted(df["school_2015"].dropna().unique()))
selected_math_teacher = st.sidebar.multiselect("Math Teacher", sorted(df["mat_teacher_1"].dropna().unique()))
selected_ela_teacher = st.sidebar.multiselect("ELA Teacher", sorted(df["ela_teacher_1"].dropna().unique()))

# Filter logic
filtered_df = df.copy()
if selected_grade:
    filtered_df = filtered_df[filtered_df["grade_2015"].isin(selected_grade)]
if selected_school:
    filtered_df = filtered_df[filtered_df["school_2015"].isin(selected_school)]
if selected_math_teacher:
    filtered_df = filtered_df[filtered_df["mat_teacher_1"].isin(selected_math_teacher)]
if selected_ela_teacher:
    filtered_df = filtered_df[filtered_df["ela_teacher_1"].isin(selected_ela_teacher)]

# Axis selectors
st.title("MAP RIT Score Explorer")
x_axis = st.selectbox("X-axis:", ["attendance_rate", "tardy_rate"])
y_axis = st.selectbox("Y-axis (RIT Score):", rit_cols)

# Plot
if not filtered_df.empty:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=filtered_df, x=x_axis, y=y_axis)
    plt.title(f"{y_axis} vs. {x_axis}")
    plt.xlabel(x_axis.replace("_", " ").title())
    plt.ylabel(y_axis.replace("_", " ").title())
    st.pyplot(plt)
else:
    st.warning("No data matches your filter selection.")
