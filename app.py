# app.py -- Week 4 Lab: GUI (Streamlit) & Problem-Solving Agents

import heapq

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Week 4 | GUI & Problem-Solving Agent",
                   layout="wide", initial_sidebar_state="expanded")

page = st.sidebar.radio("Task", [
    "1: EDA Interface",
    "2: Cloud Deployment Agent",
    "3: Course Registration Agent",
])


# ---------------------------------------------------------------- shared agent
def uniform_cost_search(graph, start, goal):
    """Cheapest start->goal path. Returns (path, cost) or (None, None)."""
    frontier = [(0, start, [start])]
    visited = set()  # prevents revisiting completed steps
    while frontier:
        cost, node, path = heapq.heappop(frontier)
        if node == goal:
            return path, cost
        if node in visited:
            continue
        visited.add(node)
        for nxt, step in graph.get(node, []):
            if nxt not in visited:
                heapq.heappush(frontier, (cost + step, nxt, path + [nxt]))
    return None, None


def render_agent(title, description, states, graph, start, goal, actions, unit):
    st.title(title)
    st.write(description)

    st.subheader("State Space")
    st.table(pd.DataFrame(
        [{"State": s, "Description": d} for s, d in states.items()]))

    st.subheader("Transition Model")
    st.table(pd.DataFrame(
        [{"From": s, "Action": actions[n], "To": n, f"Cost ({unit})": c}
         for s, edges in graph.items() for n, c in edges]))

    if st.button("Run Planning Agent"):
        path, cost = uniform_cost_search(graph, start, goal)
        if path is None:
            st.error("No plan found: goal state is unreachable.")
        else:
            st.subheader("Execution Sequence")
            st.success(" → ".join(path))
            st.table(pd.DataFrame(
                [{"Step": i, "Action": actions[nxt], "State": nxt,
                  f"Cost ({unit})": c}
                 for i, (prev, nxt) in enumerate(zip(path, path[1:]), 1)
                 for n, c in graph[prev] if n == nxt]))
            st.metric(f"Total path cost ({unit})", cost)
            st.write(f"**Goal test:** `{path[-1]}` == `{goal}` → Completed")


# ---------------------------------------------------------------------- task 1
if page.startswith("1"):
    st.title("Exploratory Data Analysis Interface")

    st.sidebar.header("Dataset Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File for Analysis",
                                             type="csv")

    if uploaded_file is None:
        st.info("Please upload a CSV file to start EDA.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not parse this file as CSV: {e}")
        st.stop()
    if df.empty or df.columns.empty:
        st.error("The uploaded CSV has no data.")
        st.stop()

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


# ---------------------------------------------------------------------- task 2
elif page.startswith("2"):
    render_agent(
        "Cloud Resource Deployment Planning Agent",
        "Initial state: deployment request received. "
        "Goal state: AI application deployed successfully.",
        states={
            "S0": "Request Received (initial)",
            "S1": "Resources Allocated",
            "S2": "VM Created",
            "S3": "Dependencies Installed",
            "S4": "AI Model Deployed",
            "S5": "Deployment Verified",
            "S6": "Deployment Completed (goal)",
        },
        graph={
            "S0": [("S1", 2)],
            "S1": [("S2", 4)],
            "S2": [("S3", 3)],
            "S3": [("S4", 5)],
            "S4": [("S5", 2)],
            "S5": [("S6", 1)],
        },
        start="S0", goal="S6",
        actions={
            "S1": "Allocate Resources",
            "S2": "Create Virtual Machine",
            "S3": "Install Dependencies",
            "S4": "Deploy AI Model",
            "S5": "Verify Deployment",
            "S6": "Mark Deployment Completed",
        },
        unit="minutes",
    )


# ---------------------------------------------------------------------- task 3
else:
    render_agent(
        "University Course Registration Planning Agent",
        "Initial state: student login. Goal state: student successfully "
        "enrolled.",
        states={
            "S0": "Student Login (initial)",
            "S1": "Student Authenticated",
            "S2": "Prerequisites Verified",
            "S3": "Courses Selected",
            "S4": "Fee Status Verified",
            "S5": "Enrollment Confirmed (goal)",
        },
        graph={
            "S0": [("S1", 1)],
            "S1": [("S2", 2)],
            "S2": [("S3", 2)],
            "S3": [("S4", 1)],
            "S4": [("S5", 1)],
        },
        start="S0", goal="S5",
        actions={
            "S1": "Authenticate Student",
            "S2": "Verify Prerequisites",
            "S3": "Select Courses",
            "S4": "Verify Fee Status",
            "S5": "Confirm Enrollment",
        },
        unit="units",
    )
