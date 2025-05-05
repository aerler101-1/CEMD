import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("student_table_enriched.csv")
    df["mat_teacher_1"] = df["mat_teacher_1"].astype(str)
    df["ela_teacher_1"] = df["ela_teacher_1"].astype(str)
    return df

df = load_data()

# Sidebar filters
st.sidebar.title("Filters")
selected_grade = st.sidebar.multiselect("Grade (2015)", sorted(df["grade_2015"].dropna().unique()))
selected_school = st.sidebar.multiselect("School (2015)", sorted(df["school_2015"].dropna().unique()))
selected_math_teacher = st.sidebar.multiselect("Math Teacher", sorted(df["mat_teacher_1"].dropna().unique()))
selected_ela_teacher = st.sidebar.multiselect("ELA Teacher", sorted(df["ela_teacher_1"].dropna().unique()))
quantile_option = st.sidebar.selectbox("Group students by test percentile quartile?", ["None", 4])
teacher_change_filter = st.sidebar.selectbox(
    "Filter by Teacher Change",
    ["All Students", "Changed Math Teacher", "Changed ELA Teacher", "Changed Either"]
)

# Apply filters
df_filtered = df.copy()
if selected_grade:
    df_filtered = df_filtered[df_filtered["grade_2015"].isin(selected_grade)]
if selected_school:
    df_filtered = df_filtered[df_filtered["school_2015"].isin(selected_school)]
if selected_math_teacher:
    df_filtered = df_filtered[df_filtered["mat_teacher_1"].isin(selected_math_teacher)]
if selected_ela_teacher:
    df_filtered = df_filtered[df_filtered["ela_teacher_1"].isin(selected_ela_teacher)]

if teacher_change_filter == "Changed Math Teacher":
    df_filtered = df_filtered[df_filtered["is_teacher_change_mat"] == True]
elif teacher_change_filter == "Changed ELA Teacher":
    df_filtered = df_filtered[df_filtered["is_teacher_change_ela"] == True]
elif teacher_change_filter == "Changed Either":
    df_filtered = df_filtered[(df_filtered["is_teacher_change_mat"] == True) | (df_filtered["is_teacher_change_ela"] == True)]

# Use most recent percentile for grouping if available
test_percentile_cols = [col for col in df_filtered.columns if col.startswith("percentile_")]
latest_percentile_col = sorted(test_percentile_cols)[-1] if test_percentile_cols else None

if quantile_option != "None" and latest_percentile_col:
    df_filtered["percentile_quantile"] = pd.qcut(df_filtered[latest_percentile_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

# Tabs for analysis
st.title("MAP Growth Analysis Dashboard")

# Tab: Growth Between Terms
growth_options = {
    "Fall → Winter 2015": ("rit_2015_Fall_", "rit_2015_Winter_"),
    "Fall → Spring 2015": ("rit_2015_Fall_", "rit_2015_Spring_"),
    "Winter → Spring 2015": ("rit_2015_Winter_", "rit_2015_Spring_"),
    "Fall 2015 → Fall 2016": ("rit_2015_Fall_", "rit_2016_Fall_"),
    "Winter 2015 → Winter 2016": ("rit_2015_Winter_", "rit_2016_Winter_"),
    "Spring 2015 → Spring 2016": ("rit_2015_Spring_", "rit_2016_Spring_")
}

st.subheader("Explore RIT Growth Between Terms")
subject = st.selectbox("Select Subject", ["mathematics", "reading"])
growth_label = st.selectbox("Select Growth Window", list(growth_options.keys()))
x_axis = st.selectbox("X-axis (Predictor)", ["attendance_rate"])

col1_prefix, col2_prefix = growth_options[growth_label]
col1 = f"{col1_prefix}{subject}"
col2 = f"{col2_prefix}{subject}"

df_filtered["growth_metric"] = df_filtered[col2] - df_filtered[col1]
df_plot = df_filtered.dropna(subset=["growth_metric", x_axis, "school_2015"])

if not df_plot.empty:
    plt.figure(figsize=(10, 6))
    if quantile_option != "None" and "percentile_quantile" in df_plot.columns:
        sns.boxplot(data=df_plot, x="percentile_quantile", y="growth_metric", hue="school_2015")
        plt.xlabel("Test Percentile Quartile")
    else:
        sns.scatterplot(data=df_plot, x=x_axis, y="growth_metric", hue="school_2015", alpha=0.6)
        plt.axhline(y=0, color="gray", linestyle="--")
        plt.xlabel(x_axis.replace("_", " ").title())
    plt.title(f"{growth_label} RIT Growth in {subject.title()} vs {x_axis.replace('_', ' ').title()}")
    plt.ylabel("RIT Growth")
    st.pyplot(plt)
else:
    st.warning("No matching data for selected growth window and subject.")

# Tab: Student Count Heatmap
st.subheader("Student Count by Grade and School")
if "grade_2015" in df.columns and "school_2015" in df.columns:
    crosstab = pd.crosstab(df_filtered["grade_2015"], df_filtered["school_2015"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(crosstab, annot=True, fmt="d", cmap="YlGnBu")
    st.pyplot(plt)

# Tab: Effectiveness Summary by Teacher
st.subheader("Fall-to-Fall Growth Effectiveness by Teacher")

teacher_palette = sns.color_palette("tab10")

# ----------------- MATH -----------------
summary_math = (
    df_filtered
    .dropna(subset=["math_growth", "ftf_2015_Fall_mathematics", "mat_teacher_1", "grade_2015"])
    .query("mat_teacher_1 != 'nan'")
    .groupby(["mat_teacher_1", "grade_2015"], as_index=False)
    .agg(
        num_students=("met_math_growth", "count"),
        pct_met_goal=("met_math_growth", "mean"),
        avg_growth=("math_growth", "mean"),
        avg_target=("ftf_2015_Fall_mathematics", "mean")
    )
    .assign(growth_above_target=lambda df: df["avg_growth"] - df["avg_target"])
    .query("num_students >= 5")
    .sort_values("pct_met_goal", ascending=True)
)

st.markdown("**Mathematics:**")
if not summary_math.empty:
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        data=summary_math,
        x="mat_teacher_1",
        y="pct_met_goal",
        hue="grade_2015",
        palette=teacher_palette,
        dodge=False
    )
    plt.ylabel("% Met Growth Goal")
    plt.xticks(rotation=45)
    plt.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)
else:
    st.info("Not enough data for math teacher summary.")

# ----------------- READING -----------------
summary_reading = (
    df_filtered
    .dropna(subset=["reading_growth", "ftf_2015_Fall_reading", "ela_teacher_1", "grade_2015"])
    .query("ela_teacher_1 != 'nan'")
    .groupby(["ela_teacher_1", "grade_2015"], as_index=False)
    .agg(
        num_students=("met_reading_growth", "count"),
        pct_met_goal=("met_reading_growth", "mean"),
        avg_growth=("reading_growth", "mean"),
        avg_target=("ftf_2015_Fall_reading", "mean")
    )
    .assign(growth_above_target=lambda df: df["avg_growth"] - df["avg_target"])
    .query("num_students >= 5")
    .sort_values("pct_met_goal", ascending=True)
)

st.markdown("**Reading:**")
if not summary_reading.empty:
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        data=summary_reading,
        x="ela_teacher_1",
        y="pct_met_goal",
        hue="grade_2015",
        palette=teacher_palette,
        dodge=False
    )
    plt.ylabel("% Met Growth Goal")
    plt.xticks(rotation=45)
    plt.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(plt)
else:
    st.info("Not enough data for reading teacher summary.")
    
    
    # ----------------- EFFECTIVENESS BY GRADE -----------------
st.subheader("Fall-to-Fall Growth Effectiveness by Grade")

summary_grade_math = (
    df_filtered
    .dropna(subset=["math_growth", "ftf_2015_Fall_mathematics", "grade_2015"])
    .groupby("grade_2015", as_index=False)
    .agg(
        num_students=("met_math_growth", "count"),
        pct_met_goal=("met_math_growth", "mean"),
        avg_growth=("math_growth", "mean"),
        avg_target=("ftf_2015_Fall_mathematics", "mean")
    )
    .assign(growth_above_target=lambda df: df["avg_growth"] - df["avg_target"])
    .query("num_students >= 5")
    .sort_values("grade_2015")
)

summary_grade_reading = (
    df_filtered
    .dropna(subset=["reading_growth", "ftf_2015_Fall_reading", "grade_2015"])
    .groupby("grade_2015", as_index=False)
    .agg(
        num_students=("met_reading_growth", "count"),
        pct_met_goal=("met_reading_growth", "mean"),
        avg_growth=("reading_growth", "mean"),
        avg_target=("ftf_2015_Fall_reading", "mean")
    )
    .assign(growth_above_target=lambda df: df["avg_growth"] - df["avg_target"])
    .query("num_students >= 5")
    .sort_values("grade_2015")
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Mathematics Effectiveness by Grade:**")
    st.dataframe(summary_grade_math.style.format({
        "pct_met_goal": "{:.0%}",
        "avg_growth": "{:.2f}",
        "avg_target": "{:.2f}",
        "growth_above_target": "{:+.2f}"
    }))

with col2:
    st.markdown("**Reading Effectiveness by Grade:**")
    st.dataframe(summary_grade_reading.style.format({
        "pct_met_goal": "{:.0%}",
        "avg_growth": "{:.2f}",
        "avg_target": "{:.2f}",
        "growth_above_target": "{:+.2f}"
    }))


# ----------------- BAR CHARTS FOR EFFECTIVENESS BY GRADE -----------------
st.markdown("### Growth Goal Achievement by Grade")

# Math Graph
if not summary_grade_math.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=summary_grade_math,
        x="grade_2015",
        y="pct_met_goal",
        palette="Blues_d"
    )
    plt.title("Math: % Students Meeting Growth Goal by Grade")
    plt.ylabel("% Met Growth Goal")
    plt.xlabel("Grade")
    plt.ylim(0, 1)
    st.pyplot(plt)
else:
    st.info("Not enough data to plot math effectiveness by grade.")

# Reading Graph
if not summary_grade_reading.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=summary_grade_reading,
        x="grade_2015",
        y="pct_met_goal",
        palette="Greens_d"
    )
    plt.title("Reading: % Students Meeting Growth Goal by Grade")
    plt.ylabel("% Met Growth Goal")
    plt.xlabel("Grade")
    plt.ylim(0, 1)
    st.pyplot(plt)
else:
    st.info("Not enough data to plot reading effectiveness by grade.")


