import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("data/focus_data.csv")

# 2. Select features (inputs)
X = df[
    [
        "session_duration",
        "pause_count",
        "task_switch_count",
        "completion_rate"
    ]
]

# 3. Select label (output)
y = df["label"]

# 4. Split into training & testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Create model
model = LogisticRegression()

# 6. Train model
model.fit(X_train, y_train)

# 7. Test model
predictions = model.predict(X_test)

# 8. Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)
print("✅ Model accuracy:", accuracy)

new_session = pd.DataFrame(
    [[20, 3, 5, 0.4]],
    columns=[
        "session_duration",
        "pause_count",
        "task_switch_count",
        "completion_rate"
    ]
)


# Get probability
probability = model.predict_proba(new_session)[0][1]

focus_score = (1 - probability) * 100
print(f"🧠 Focus Score: {focus_score:.1f}%")

if focus_score < 50:
    print("⚠️ High task switching tendency")
else:
    print("✅ Good focus pattern")
print("\n--- Enter session details ---")

session_duration = float(input("Session duration (minutes): "))
pause_count = int(input("Number of pauses: "))
task_switch_count = int(input("Number of task switches: "))
completion_rate = float(input("Completion rate (0 to 1): "))

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

print(f"\n🧠 Focus Score: {focus_score:.1f}%")

if focus_score < 50:
    print("⚠️ High task switching tendency")
else:
    print("✅ Good focus pattern")
