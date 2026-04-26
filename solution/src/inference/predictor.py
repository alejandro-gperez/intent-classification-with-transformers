import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import json

from .config import CONFIDENCE_THRESHOLD, GAP_THRESHOLD
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


# ===== HELPER =====
def map_label_to_category(label: str) -> str:
    for category, labels in categories.items():
        if label in labels:
            return category
    return "unknown"


# ===== MAIN FUNCTION =====
def predict(text: str) -> dict:
    """
    Predict intent using transformer + safety net logic.
    """

    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = F.softmax(logits, dim=1).squeeze()

    top2 = torch.topk(probs, k=2)

    top1_idx = top2.indices[0].item()
    top2_idx = top2.indices[1].item()

    top1_conf = top2.values[0].item()
    top2_conf = top2.values[1].item()

    gap = top1_conf - top2_conf

    # ===== SAFETY NET =====
    if top1_conf >= CONFIDENCE_THRESHOLD:
        final_idx = top1_idx
    elif gap < GAP_THRESHOLD:
        final_idx = top2_idx
    else:
        final_idx = top1_idx

    # ===== DECODE =====
    final_label = label_encoder[final_idx]
    category = map_label_to_category(final_label)

    return {
        "label": final_label,
        "category": category,
        "confidence": round(top1_conf, 4)
    }