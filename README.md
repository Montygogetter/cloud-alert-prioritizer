# Cloud Alert Prioritizer – AI with NLP, Classification, Search, and Symbolic Reasoning

This project simulates an AI-driven alert analyzer for cloud operations. It classifies alert severity using NLP and logistic regression, prioritizes them using best-first search, and applies first-order logic rules to recommend escalation, suppression, or closure actions.

---

## 🧠 Key Features

- 🔹 Natural Language Processing using TF-IDF
- 🔹 Logistic Regression classifier with class balancing
- 🔹 Best-First Search heuristic using urgency and system weighting
- 🔹 First-Order Logic rules for symbolic decision-making
- 🔹 CSV output for top 10 ranked alerts with automated decisions

---

## 📂 Files

- `CSC510_Module8_PortfolioProject_Code_AI_AlertAnalyzer_Jessica_Montgomery.py` – full implementation
- `top_10_with_fol_decisions.csv` – sample output showing alert decisions

---

## 🚀 How to Run

1. Place your expanded alert dataset as `alert_dataset_expanded.csv` in the same directory
2. Run:
```bash
python CSC510_Module8_PortfolioProject_Code_AI_AlertAnalyzer_Jessica_Montgomery.py
