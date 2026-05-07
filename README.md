# Reto Analítico Banco Agromercantil 
## Executive Report - Group AAC
Group Members: Alejandra Sierra, Alejandro Pérez, Camila Sandoval

---

## Project Summary

The bank receives thousands of messages daily through different channels: app, web, email, and social media. Before this system, a team of analysts read them one by one and manually redirected them to the correct area. That process does not scale, causes delays, and does not make good use of the people’s time.

This project builds an automatic classifier that reads each message, identifies the customer’s intent among 77 possible ones, and sends it to the correct queue in a matter of seconds. Everything runs inside Docker containers connected to RabbitMQ, without relying on external services.

---

## From cleaning to production: how we got here

The work was divided into three parts that were built on top of each other:

**Part 1 — Data and model.** We started by exploring the dataset: we checked nulls, duplicates, class distribution, and overall text quality. With that clear, we prepared the data for training: we encoded the labels, performed a stratified train-validation split to preserve the distribution of the 77 classes, tokenized the texts using the DistilBERT tokenizer, and computed class weights to compensate for imbalance. Then we fine-tuned the model and evaluated it using F1.

**Part 2 — Prediction logic.** CWith the trained model, the next step was to turn it into something robust for production. A `predict()` function was built that not only returns the most probable label, but also evaluates how confident the model is before making a decision. If confidence is low or the top two options are very close, the system flags it for review instead of guessing.

**Part 3 — Worker and integration.** The final piece connects everything: a dockerized worker that consumes messages from RabbitMQ, calls `predict()`, and publishes the result to the queue of the corresponding category. The full stack can be brought up with a single command.

---

## Why DistilBERT

Evaluamos tres enfoques: un modelo de machine learning tradicional, fine-tuning de un transformer, y un flujo con LLM. La decisión fue DistilBERT por varias razones concretas:

| Criteria | Traditional ML | DistilBERT (chosen) | LLM workflow |
|---|---|---|---|
| Expected F1 (77 classes) | 0.70–0.80 | **0.85–0.90** | 0.88–0.92 |
| Latency per message | <10ms | ~80ms CPU / ~15ms GPU | 500ms–2s |
| Operating cost | Low | Low (local model) | High (External API) |
| Red independency | ✅ | ✅ | ❌ |
| Runs in CPU (Docker) | ✅ | ✅ | Depends |

DistilBERT is a compact version of BERT that maintains good accuracy but runs well on CPU, which was important because the worker runs inside Docker without any GPU guarantee. In addition, Hugging Face provides the full ecosystem ready to use — tokenizer, pretrained model, Trainer — so within the available time (5 days) it was the most viable option to implement correctly.

An LLM would have given a marginally higher F1, but it depends on a paid external API, has variable latency, and introduces a network dependency that complicates local deployment. Traditional ML does not reach the required F1 with 77 imbalanced classes.

An additional advantage: class imbalance was handled using `class_weights` directly in the loss function, something DistilBERT allows to be done cleanly and that in traditional models usually requires more complex techniques like oversampling or SMOTE.

**Obtained result:** F1 weighted **0.8539** · F1 macro **0.8557** · Accuracy **0.86** over 2,064 validation examples.

---

## Training Pipeline 

The data preparation process was just as important as the training itself. DistilBERT does not understand raw text directly — it needs each message to be converted into numerical sequences (tokens) with a specific format. Before getting there, the dataset had to be clean.

Nulls and duplicates were removed to avoid introducing noise into the training. Then, the 77 labels were encoded as numbers, a stratified split was performed (80% training, 20% validation) so that all classes were represented in both partitions, and the texts were tokenized with `MAX_LENGTH=256`, which is enough to cover most customer messages without truncating relevant information.

Class imbalance was evident: some intents had many more examples than others. Without correction, the model tends to ignore smaller classes. We computed weights inversely proportional to the frequency of each class and applied them in the loss function during training.
**PProgression by epoch:**

| Epoch | Training Loss | Validation Loss | F1 Weighted | F1 Macro |
|---|---|---|---|---|
| 1 | 3.2958 | 2.9744 | 0.4650 | 0.4720 |
| 2 | 2.0829 | 1.8355 | 0.7561 | 0.7591 |
| 3 | 1.4419 | 1.2500 | 0.8224 | 0.8238 |
| 4 | 1.0858 | 0.9856 | 0.8517 | 0.8527 |
| 5 | 0.9481 | 0.9094 | **0.8539** | **0.8557** |

The model improved consistently without signs of overfitting — the validation loss decreased in every epoch.

---

## Prediction logic & safety net

Once the model was trained, the priority was no longer improving F1, but making it usable in production.

The model returns a probability distribution over 77 classes. In many cases, the correct prediction is not accompanied by high confidence. This does not mean the model is wrong, it means the message is ambiguous in nature. In other words, there are many similar intents and the language used by users is not 100% precise.

If we only take the `argmax()` prediction, we force the system to make a decision even when it is not confident in its answer. In a production environment, this introduces errors that are difficult to detect later.

Given this issue, an additional layer was implemented in the decision logic: a Safety Net based on model confidence and probabilities. All of this is contained within `predict()` and uses `config.py` and `preprocessing.py`.

### How does `predict()` work

1. `preprocessing.py` Cleans the input text.
2. It is tokenized using the model’s tokenizer.
3. Runs inference with DistilBERT.
4. Converts logits to probabilities (softmax).
5. Extracts the two most probable predictions (top-1 and top-2).
6. Applies a decision logic based on:
   - absolute confidence of the top-1
   - difference between top-1 and top-2 (gap)

### Safety Net

There are two key signals in a model’s output:
- Confidence: how sure it is about its prediction.
- Gap (top1 - top 2): how clear the difference is between options.

So, in the following case, it would be incorrect to choose top-1 arbitrarily since the model is not confident between both responses:

```bash
Top-1: pending_transfer → 0.48
Top-2: transfer_not_received → 0.45
Gap: 0.03
```
### Final decision logic

To fix this, 3 zones were implemented: `high_confidence`, `medium_confidence`, `low_confidence` used the following way:

```bash
if confidence ≥ 0.65:
    → high_confidence
    → trust top-1

if 0.45 ≤ confidence < 0.65:
    if gap < 0.15:
        → ambiguous_used_top2
        → use top-2 (real ambiguity)
    else:
        → medium_confidence
        → use top-1

if confidence < 0.45:
    → low_confidence
    → use top-1 and mark as uncertain
```

The configuration of this logic was implemented in `config.py` to provide a separate control panel for calibration or future adjustments.

Just as always using top-1 is a mistake, always using top-2 is as well. This was discovered while calibrating `predict()`. As the use of top-2 increased, and the gap threshold was raised, the result was more overrides and decreased precision. Forcing the use of top-2 introduces more errors than it fixes.

The need for the Safety Net lies in the fact that without this layer, the model assumes it is always right. By implementing it, the model:
- Identifies how confident it is
- Detects ambiguity
- Avoids making arbitrary decisions
- Exposes uncertainty for monitoring

In other words, the system not only predicts, but also evaluates the quality of its own prediction before acting.

### Results after calibrating

The system was evaluated on a validation subset (1,000 examples):
- Accuracy: 0.88
- High confidence: 26%
- Medium confidence: 38%
- Low confidence: 36%

This confirms that the model is often correct, but not always with high confidence. The system is able to recognize this uncertainty.

During testing, it was found that the model makes most of its errors in pairs that are semantically close:
- `top_up_failed` and `pending_top_up`
- `declined_transfer` and `declined_card_payment`
- `verify_my_identity` and `why_verify_identity`

Even for a human, these cases can be ambiguous. The model is able to reflect that difficulty and does not hide it thanks to the Safety Net.

### `predict()` output

Each message includes additional useful information, with the goal of traceability:

```json
{
  "label": "pending_transfer",
  "category": "transferencias",
  "confidence": 0.48,
  "top2_label": "transfer_not_received_by_recipient",
  "gap": 0.03,
  "decision": "low_confidence"
}
```

This is intended to make debugging easier, enable monitoring, prioritize human review, and support further analysis of the model.

---

## Where does the mode live

The model lives inside the Docker image. When the image is built, the model files are copied inside and the worker loads them at startup. It does not need to download anything or connect to any external service to function.

```
Imagen Docker
└── /app/
    ├── worker.py
    ├── src/inference/
    │   ├── predictor.py       ← predict() function with safety net
    │   ├── config.py          ← confidence thresholds
    │   └── preprocessing.py   ← string cleaning
    └── model/
        ├── model.safetensors  ← model weights (~250MB)
        ├── tokenizer.json
        ├── config.json
        └── label_encoder.pkl  ← mapping label_id → intent name
```

This guarantees that the same image always uses the same model. The limitation is that if the model is retrained, the image must be rebuilt. In a more mature production environment, a registry such as MLflow or S3 would typically be used for versioning and artifact storage, but for the scope of this project, the embedded solution is the correct choice.

---

## How to update the model

If new data is obtained or the model needs to be improved, the process is:

1. Run `cleaning/model-training.ipynb` with the updated data. This generates new files in `model/`.
2. Rebuild and restart the worker:

```bash
docker compose -f deploy/compose.yml -f solution/compose.override.yml build worker
docker compose -f deploy/compose.yml -f solution/compose.override.yml up -d worker
```

While the worker restarts, incoming messages are not lost — they remain queued in `incoming_social_messages` and are processed as soon as the new worker is ready. This is a direct advantage of the event-driven architecture: the classifier and the input channel are decoupled.

---

## What happens if the message volume doubles

The current worker processes one message at a time. If the volume grows significantly, messages accumulate in the queue — RabbitMQ handles them without issue — but the response time increases.

The solution is to spin up more worker instances:

```bash
docker compose -f deploy/compose.yml -f solution/compose.override.yml up -d --scale worker=3
```

With that, three workers process messages in parallel from the same queue, and RabbitMQ distributes them automatically. No code changes are required. The bottleneck is model inference (not the broker), and adding workers is the most direct way to address it.

Each DistilBERT instance uses approximately 250MB of memory. With three instances, that’s about ~750MB, which is manageable on any modern server.

---

## How is it monitored in production?

**RabbitMQ Management UI** (`http://localhost:15672`): allows you to see in real time how many messages are in each queue, the processing rate, and whether the worker is active. If the input queue keeps growing without stopping, something is wrong with the worker.

**Worker logs:** each message leaves a record with its label, category, confidence, and decision type:

```bash
# See messages with low confidence
docker logs bam_worker -f | grep "low_confidence"
```

**Category distribution:** if suddenly 80% of messages are going to `fraude` when historically it was 15%, there’s a problem — either with the model or with the label mapping.

| Cue | Probable cause |
|---|---|
| Input queue keeps growing without stopping | Worker down or too slow |
| Many messages with `low_confidence` | Messages outside what the model knows |
| One category receives almost all the traffic | Bug in the label → category mapping |
| Low F1 in human review | The model needs to be retrained |

The metric that truly matters to the business is not the F1 score in the notebook, but how many messages reach the correct team. That can only be measured by comparing predictions against human reviews in production.

---

## Bring up the full stack

```bash
docker compose -f deploy/compose.yml -f solution/compose.override.yml up -d
```

That command starts RabbitMQ, the synthetic message producer, and the worker. Messages begin to flow automatically into the category queues.
