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


#----- tab 6: ftf -----

tab6 = st.tabs(["FTF Growth by Teacher"])[0]

with tab6:
    st.subheader("Fall-to-Fall Growth Effectiveness by Math Teacher")

    # Drop missing values
    df_clean = df.dropna(subset=[
        "math_growth", "ftf_2015_Fall_mathematics", "met_math_growth", "mat_teacher_1"
    ])

    # Group by teacher
    summary = (
        df_clean
        .groupby("mat_teacher_1")
        .agg(
            num_students=("met_math_growth", "count"),
            pct_met_goal=("met_math_growth", "mean"),
            avg_growth=("math_growth", "mean"),
            avg_target=("ftf_2015_Fall_mathematics", "mean")
        )
        .assign(growth_above_target=lambda d: d["avg_growth"] - d["avg_target"])
        .query("num_students >= 5")
        .sort_values("pct_met_goal", ascending=False)
    )

    # Select display metric
    display_metric = st.selectbox(
        "Choose Metric to Rank Teachers",
        ["pct_met_goal", "growth_above_target", "avg_growth", "avg_target"]
    )

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary.reset_index().sort_values(display_metric, ascending=False),
        x="mat_teacher_1",
        y=display_metric
    )
    plt.ylabel(display_metric.replace("_", " ").title())
    plt.xlabel("Math Teacher")
    plt.title(f"Math Teachers Ranked by {display_metric.replace('_', ' ').title()}")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Optional: show table
    st.markdown("### Summary Table")
    st.dataframe(summary.round(2).reset_index())
    
    
    # ---------- Tab 7: Explore Growth Across Terms ----------
tab7 = st.tabs(["Explore RIT Growth Across Terms"])[0]

with tab7:
    st.subheader("Explore MAP Growth Between Terms")

    growth_options = {
        "Fall 2015 → Fall 2016": ("rit_2015_Fall_", "rit_2016_Fall_"),
        "Fall 2015 → Winter 2015": ("rit_2015_Fall_", "rit_2015_Winter_"),
        "Winter 2015 → Winter 2016": ("rit_2015_Winter_", "rit_2016_Winter_"),
    }

    subject = st.selectbox("Select Subject", ["mathematics", "reading"])
    growth_label = st.selectbox("Select Growth Window", list(growth_options.keys()))
    x_axis = st.selectbox("X-axis (Predictor)", ["attendance_rate", "tardy_rate"])

    col1_prefix, col2_prefix = growth_options[growth_label]
    col1 = f"{col1_prefix}{subject}"
    col2 = f"{col2_prefix}{subject}"

    df["growth_metric"] = df[col2] - df[col1]

    df_plot = df.dropna(subset=["growth_metric", x_axis, "school_2015"])

    if not df_plot.empty:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df_plot, x=x_axis, y="growth_metric", hue="school_2015", alpha=0.6)
        plt.axhline(y=0, color="gray", linestyle="--")
        plt.title(f"{growth_label} RIT Growth in {subject.title()} vs {x_axis.replace('_', ' ').title()}")
        plt.xlabel(x_axis.replace("_", " ").title())
        plt.ylabel("RIT Growth")
        st.pyplot(plt)
    else:
        st.warning("No matching data for selected growth window and subject.")

