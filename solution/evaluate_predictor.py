import os
import pandas as pd
from collections import Counter

from src.inference.predictor import predict

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data/train_clean.csv"))

# ===== LOAD DATA =====
df = pd.read_csv(DATA_PATH)

# Optional: limit for faster testing
df = df.sample(n=1000, random_state=42)  # comment this out for full dataset

# ===== METRICS =====
correct = 0
total = 0



decision_counts = Counter()
confidence_values = []
wrong = []

# ===== EVALUATION LOOP =====
for _, row in df.iterrows():
    text = row["text"]
    true_label = row["label"]

    result = predict(text)

    pred_label = result["label"]
    decision = result["decision"]
    confidence = result["confidence"]

    if pred_label == true_label:
        correct += 1
    else:
        wrong.append((text, pred_label, true_label, confidence))


    decision_counts[decision] += 1
    confidence_values.append(confidence)

    total += 1

# ===== RESULTS =====
accuracy = correct / total

print("\n===== EVALUATION RESULTS =====")
print(f"Samples evaluated: {total}")
print(f"Accuracy: {accuracy:.4f}")

print("\n===== DECISION DISTRIBUTION =====")
for k, v in decision_counts.items():
    print(f"{k}: {v} ({v/total:.2%})")

print("\n===== CONFIDENCE STATS =====")
print(f"Average confidence: {sum(confidence_values)/len(confidence_values):.4f}")
print(f"Max confidence: {max(confidence_values):.4f}")
print(f"Min confidence: {min(confidence_values):.4f}")

print("\n===== SAMPLE ERRORS =====")
for w in wrong[:50]:
    print(w)