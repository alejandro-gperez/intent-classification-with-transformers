from src.inference.predictor import predict

tests = [
    "I lost my card",
    "My payment was declined",
    "Why is my transfer pending?",
    "I got charged twice",
    "How do I verify my identity?"
]

for text in tests:
    print("\nINPUT:", text)
    print("OUTPUT:", predict(text))