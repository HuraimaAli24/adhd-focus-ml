import pandas as pd
import numpy as np

np.random.seed(42)

sessions = []

for _ in range(500):
    session_duration = np.random.randint(5, 60)  # minutes
    pause_count = np.random.randint(0, 10)
    task_switch_count = np.random.randint(0, 6)
    completion_rate = np.random.uniform(0.3, 1.0)

    # Label rule (simulated behavior)
    if task_switch_count >= 4 and session_duration < 25:
        label = 1  # High task switching
    else:
        label = 0  # Focused

    sessions.append([
        session_duration,
        pause_count,
        task_switch_count,
        completion_rate,
        label
    ])

df = pd.DataFrame(
    sessions,
    columns=[
        "session_duration",
        "pause_count",
        "task_switch_count",
        "completion_rate",
        "label"
    ]
)

df.to_csv("data/focus_data.csv", index=False)
print("✅ Dataset created: data/focus_data.csv")
