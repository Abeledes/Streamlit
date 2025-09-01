import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import altair as alt

# Mock LLM explainer
def explain_review(review_text, risk):
    if risk > 0.85:
        return {
            "explanation": f"Review '{review_text[:50]}...' looks highly suspicious: repetitive language or generic praise.",
            "recommended_action": "BLOCK_AND_REVIEW"
        }
    elif risk > 0.6:
        return {
            "explanation": f"Review '{review_text[:50]}...' has some suspicious patterns.",
            "recommended_action": "SHADOW_AND_REVIEW"
        }
    else:
        return {
            "explanation": "Review appears normal.",
            "recommended_action": "PUBLISH"
        }

# Feature extraction
def extract_features(df):
    df["length"] = df["text"].str.len()
    df["exclam"] = df["text"].str.count("!")
    df["sentiment"] = df["text"].apply(lambda x: TextBlob(x).sentiment.polarity)
    # Optional: simulate date for dashboard charts
    if "created_at" not in df.columns:
        df["created_at"] = pd.date_range(end=pd.Timestamp.today(), periods=len(df))
    return df

# Risk scoring simulation
def compute_risk(df):
    np.random.seed(42)
    df["rcf_score"] = np.random.rand(len(df))
    df["xgb_score"] = np.random.rand(len(df))
    df["risk"] = 0.5 * df["rcf_score"] + 0.5 * df["xgb_score"]
    df["action"] = df["risk"].apply(
        lambda r: "BLOCK_AND_REVIEW" if r > 0.85 else ("SHADOW_AND_REVIEW" if r > 0.6 else "PUBLISH")
    )
    df["llm_output"] = df.apply(lambda row: explain_review(row["text"], row["risk"]), axis=1)
    return df

# Streamlit UI
st.title("Trusted Shops Fake Review Detection Simulator")

uploaded_file = st.file_uploader("Upload CSV with reviews", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = extract_features(df)
    df = compute_risk(df)

    st.subheader("Review Decisions Table")
    df_display = df[["review_id", "text", "rating", "risk", "action", "llm_output"]]
    st.dataframe(df_display)

    # Summary metrics
    st.subheader("Summary Metrics")
    st.write("Total Reviews:", len(df))
    st.write("Blocked Reviews:", (df["action"] == "BLOCK_AND_REVIEW").sum())
    st.write("Shadowed Reviews:", (df["action"] == "SHADOW_AND_REVIEW").sum())
    st.write("Published Reviews:", (df["action"] == "PUBLISH").sum())

    # Dashboard: Bar chart of actions
    st.subheader("Review Actions Distribution")
    action_counts = df.groupby("action").size().reset_index(name="count")
    chart = alt.Chart(action_counts).mark_bar().encode(
        x="action",
        y="count",
        color="action"
    )
    st.altair_chart(chart, use_container_width=True)

    # Dashboard: Actions over time
    st.subheader("Review Actions Over Time")
    df_time = df.groupby([df["created_at"].dt.date, "action"]).size().reset_index(name="count")
    chart_time = alt.Chart(df_time).mark_line(point=True).encode(
        x="created_at:T",
        y="count:Q",
        color="action:N",
        tooltip=["created_at:T", "action:N", "count:Q"]
    )
    st.altair_chart(chart_time, use_container_width=True)
