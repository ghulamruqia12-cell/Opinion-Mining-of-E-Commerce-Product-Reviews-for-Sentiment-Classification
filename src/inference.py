"""
Lightweight CLI for running inference on unseen text using saved artifacts.

Usage:
    python -m src.inference --text "Great battery life" "Terrible sound quality"
    python -m src.inference --interactive
"""

import argparse
from pathlib import Path
from joblib import load

from utils.config_loader import load_config
from src.features import transform_texts


def load_artifacts(config):
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    word_path = artifacts_dir / "tfidf_word.joblib"
    char_path = artifacts_dir / "tfidf_char.joblib"
    model_path = artifacts_dir / "svm_model.joblib"

    if not (word_path.exists() and char_path.exists() and model_path.exists()):
        raise FileNotFoundError(
            f"Artifacts not found in {artifacts_dir.resolve()}. "
            "Run `python -m src.pipeline` first to train and save them."
        )

    word_vec = load(word_path)
    char_vec = load(char_path)
    model = load(model_path)
    return model, word_vec, char_vec


def predict_texts(model, word_vec, char_vec, texts):
    features = transform_texts(texts, word_vec, char_vec)
    return model.predict(features)


def main():
    parser = argparse.ArgumentParser(description="Run sentiment inference from console.")
    parser.add_argument(
        "--text",
        nargs="*",
        help="One or more review strings to classify. If omitted, uses demo texts.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="If set, read lines from stdin until an empty line is entered.",
    )
    args = parser.parse_args()

    config = load_config()
    model, word_vec, char_vec = load_artifacts(config)

    inputs = args.text or [
        "Packaging was fine but the product feels cheap.",
        "Absolutely love this laptop, fast delivery and great build quality.",
        "Battery died in two days and support never replied.",
    ]

    if args.interactive:
        print("Type a review and press Enter (empty line to stop):")
        while True:
            line = input("> ").strip()
            if not line:
                break
            inputs.append(line)

    preds = predict_texts(model, word_vec, char_vec, inputs)
    for review, label in zip(inputs, preds):
        print(f"{label}: {review}")


if __name__ == "__main__":
    main()
