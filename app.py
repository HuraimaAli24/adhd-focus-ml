import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/focus_data.csv")

X = df[
    [
        "session_duration",
        "pause_count",
        "task_switch_count",
        "completion_rate"
    ]
]
y = df["label"]

# Train model
model = LogisticRegression()
model.fit(X, y)

st.title("🧠 Focus Pattern Analyzer")
st.write("This app analyzes session behavior to infer focus patterns.")
st.write("⚠️ This is NOT a medical diagnostic tool.")

st.header("Enter session details")

session_duration = st.slider("Session duration (minutes)", 5, 60, 25)
pause_count = st.slider("Number of pauses", 0, 10, 2)
task_switch_count = st.slider("Task switches", 0, 6, 2)
completion_rate = st.slider("Completion rate", 0.0, 1.0, 0.7)

if st.button("Analyze Focus"):
    user_session = pd.DataFrame(
        [[session_duration, pause_count, task_switch_count, completion_rate]],
        columns=[
            "session_duration",
            "pause_count",
            "task_switch_count",
            "completion_rate"
        ]
    )

    probability = model.predict_proba(user_session)[0][1]
    focus_score = (1 - probability) * 100

    st.subheader(f"🧠 Focus Score: {focus_score:.1f}%")

    if focus_score < 50:
        st.warning("High task switching tendency")
    else:
        st.success("Good focus pattern detected")

