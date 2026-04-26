import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json

from .config import CONFIDENCE_THRESHOLD, MEDIUM_CONFIDENCE_THRESHOLD, GAP_THRESHOLD
from .preprocessing import clean_text

# ===== LOAD MODEL ONCE (IMPORTANT) =====
BASE_DIR = os.path.dirname(__file__)

MODEL_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "../../../model")
)

MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../model"))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open(f"{DATA_DIR}/categories.json", "r") as f:
    categories = json.load(f)


# ===== HELPER ===== #
def map_label_to_category(label: str) -> str:
    for category, labels in categories.items():
        if label in labels:
            return category
    return "unknown"


# ===== DECISION LOGIC ===== #
def apply_decision_logic(top1_idx, top2_idx, top1_conf, top2_conf):
    """
    Decide final prediction based on confidence and gap.

    Returns:
        final_idx (int)
        decision (str)
        gap (float)
    """

    gap = top1_conf - top2_conf

    if top1_conf >= CONFIDENCE_THRESHOLD:
        return top1_idx, "high_confidence", gap

    elif top1_conf >= MEDIUM_CONFIDENCE_THRESHOLD:
        if gap < GAP_THRESHOLD:
            return top2_idx, "ambiguous_used_top2", gap
        else:
            return top1_idx, "medium_confidence", gap

    else:
        return top1_idx, "low_confidence", gap


# ===== MAIN FUNCTION =====
def predict(text: str) -> dict:
    """
    Predict intent using transformer + structured safety net.

    Returns:
        dict with:
            - label (final decision)
            - category
            - confidence (top-1)
            - top1_label
            - top2_label
            - top2_confidence
            - gap (top1 - top2)
            - decision (reasoning tag)
    """

    # ===== PREPROCESS =====
    text = clean_text(text)

    # ===== TOKENIZE =====
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # ===== MODEL INFERENCE =====
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1).squeeze()

    # ===== TOP-K =====
    top2 = torch.topk(probs, k=2)

    top1_idx = top2.indices[0].item()
    top2_idx = top2.indices[1].item()

    top1_conf = top2.values[0].item()
    top2_conf = top2.values[1].item()

    # ===== DECISION LAYER =====
    final_idx, decision, gap = apply_decision_logic(
        top1_idx,
        top2_idx,
        top1_conf,
        top2_conf
    )

    # ===== LABEL DECODE =====
    final_label = label_encoder.get(final_idx, "unknown")
    top1_label = label_encoder.get(top1_idx, "unknown")
    top2_label = label_encoder.get(top2_idx, "unknown")

    # ===== CATEGORY MAP =====
    category = map_label_to_category(final_label)

    # ===== DEBUGGING LOGS, DISABLE FOR DEADLINE =====#
    #print(f"[DEBUG] text='{text}' | top1={top1_conf:.3f} | top2={top2_conf:.3f} | gap={gap:.3f} | decision={decision}")

    # ===== OUTPUT =====
    return {
        "label": final_label,
        "category": category,
        "confidence": round(top1_conf, 4),
        "top1_label": top1_label,
        "top2_label": top2_label,
        "top2_confidence": round(top2_conf, 4),
        "gap": round(gap, 4),
        "decision": decision
    
    }