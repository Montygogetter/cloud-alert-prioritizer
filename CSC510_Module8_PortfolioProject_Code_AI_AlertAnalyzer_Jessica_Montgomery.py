"""
CSC510 – Foundations of Artificial Intelligence
Portfolio Project – Final Submission
AI Alert Analyzer with NLP, Classification, Search, and FOL Reasoning

Author: Jessica Montgomery
Instructor: Dr. Gonzalez
Date: June 2, 2025

Description:
This program simulates an AI-powered alert analyzer for cloud operations. It uses natural language
processing (TF-IDF) and a logistic regression classifier to assess the severity of system alerts.
Alerts are prioritized using a Best-First Search heuristic based on severity, system criticality,
and urgency keywords. A symbolic expert system applies first-order logic rules to escalate,
suppress, or auto-close alerts.

Modules:
1. Data Loading and Preparation
2. NLP Vectorization and Classification
3. Best-First Search Prioritization
4. First-Order Logic-Based Decision System

Output:
- Confusion matrix and classification report for model evaluation
- Sample predictions on test alerts
- Ranked top 10 prioritized alerts
- CSV output with FOL-based decisions

"""


# Module 1: Load and Prepare Alert Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Expanded Dataset
df = pd.read_csv(r"C:\Users\Jessica Montgomery\OneDrive\CSU Global Homework\CSC510 Foundations of Artificial Intelligence\alert_dataset_expanded.csv")  
# Step 2: Preprocessing
X_text = df["message"]
y = df["historical_severity"]

# Module 2: NLP Feature Extraction and Classification
# Step 3: Vectorize Text
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X_text)

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# Step 5: Train Classifier with Class Weight Balancing
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 7: Predict on New Alerts
new_alerts = ["Disk usage at 95%", "Scheduled system backup", "Multiple failed login attempts"]
new_vectors = vectorizer.transform(new_alerts)
predictions = model.predict(new_vectors)

for msg, pred in zip(new_alerts, predictions):
    print(f"\n'{msg}' predicted as: {pred}")

# Module 3: Best-First Search Prioritization 
import heapq

# Define priority rules
severity_priority = {"critical": 0, "informational": 1, "false_positive": 2}
system_priority = {"database": 0, "backend": 0, "network": 0,
                   "frontend": 1, "email": 1, "storage": 1}
urgency_keywords = ["failed", "exceeds", "shutdown", "lost", "error", "latency", "delayed"]

# Heuristic scoring function
def compute_priority(alert_row, predicted_severity):
    sev_score = severity_priority.get(predicted_severity, 2)
    sys_score = system_priority.get(alert_row["system"], 1)
    msg = alert_row["message"].lower()
    keyword_bonus = -1 if any(word in msg for word in urgency_keywords) else 0
    return sev_score + sys_score + keyword_bonus

# Predict severity for all alerts
df["predicted_severity"] = model.predict(vectorizer.transform(df["message"]))

# Create aectorizer.transform(df["message"]))

# Create priority queue
priority_queue = []
for idx, row in df.iterrows():
    score = compute_priority(row, row["predicted_severity"])
    heapq.heappush(priority_queue, (score, idx, row.to_dict()))

# Extract top 10 alerts
top_alerts = []
for _ in range(10):
    if priority_queue:
        score, _, alert = heapq.heappop(priority_queue)
        alert["priority_score"] = score
        top_alerts.append(alert)

# Create top_df for FOL processing
top_df = pd.DataFrame(top_alerts)

# Module 4: First-Order Logic (FOL) Rules 
def apply_fol_rules(alert):
    severity = alert["predicted_severity"]
    system = alert["system"].lower()
    message = alert["message"].lower()

    if severity == "critical" and system in ["database", "backend"]:
        return "ESCALATE: Notify on-call engineer"
    elif severity == "informational" and any(word in message for word in ["maintenance", "backup"]):
        return "SUPPRESS: Routine informational"
    elif severity == "false_positive":
        return "AUTO-CLOSE: No action needed"
    else:
        return "REVIEW: Manual inspection required"

# Apply rules
top_df["FOL_decision"] = top_df.apply(apply_fol_rules, axis=1)

# Save final results
top_df.to_csv("top_10_with_fol_decisions.csv", index=False)

# Preview output
print("\n📋 Top 10 Alerts with FOL-Based Decisions:")
print(top_df[["message", "system", "predicted_severity", "priority_score", "FOL_decision"]])
