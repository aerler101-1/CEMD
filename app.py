import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("student_table_enriched.csv")

df = load_data()

# Setup
st.title("MAP Student Dashboard")
st.markdown("Visualize growth, attendance, and behavior patterns by student, teacher, and school.")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "RIT vs Attendance/Tardy",
    "Tardy Rate by School",
    "Growth Rate by Teacher",
    "RIT Growth Distribution",
    "Grade x School Heatmap"
])

# ---------- Tab 1: Scatterplot ----------
with tab1:
    st.subheader("Scatter: Any RIT vs Attendance or Tardy Rate")
    rit_cols = [col for col in df.columns if col.startswith("rit_")]
    x_col = st.selectbox("X-axis", ["attendance_rate", "tardy_rate"])
    y_col = st.selectbox("Y-axis (RIT score)", rit_cols)

    if not df.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue="school_2015", alpha=0.6)
        plt.title(f"{y_col} vs. {x_col}")
        st.pyplot(plt)

# ---------- Tab 2: Boxplot ----------
with tab2:
    st.subheader("Tardy Rate Distribution by School")
    if "tardy_rate" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="school_2015", y="tardy_rate")
        plt.xticks(rotation=45)
        st.pyplot(plt)

# ---------- Tab 3: Bar Chart by Teacher ----------
with tab3:
    st.subheader("Percent of Students Meeting Math Growth by Teacher")
    if "met_math_growth" in df.columns and "mat_teacher_1" in df.columns:
        growth_by_teacher = (
            df.groupby("mat_teacher_1")["met_math_growth"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        plt.figure(figsize=(10, 6))
        sns.barplot(data=growth_by_teacher, x="mat_teacher_1", y="met_math_growth")
        plt.ylabel("Proportion Meeting Growth")
        plt.xlabel("Math Teacher")
        st.pyplot(plt)

# ---------- Tab 4: Histogram ----------
with tab4:
    st.subheader("Distribution of Math RIT Growth")
    if "math_growth" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df["math_growth"].dropna(), kde=True, bins=30)
        plt.axvline(x=0, color="red", linestyle="--")
        st.pyplot(plt)

# ---------- Tab 5: Crosstab Heatmap ----------
with tab5:
    st.subheader("Student Count by Grade and School")
    if "grade_2015" in df.columns and "school_2015" in df.columns:
        crosstab = pd.crosstab(df["grade_2015"], df["school_2015"])
        plt.figure(figsize=(10, 6))
        sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu")
        st.pyplot(plt)
