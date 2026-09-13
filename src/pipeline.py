"""
End-to-end training script.

Usage:
    python -m src.pipeline
"""

from pathlib import Path
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

from utils.config_loader import load_config
from utils.visualization import (
    plot_rating_distribution,
    plot_class_distribution,
    plot_wordcloud,
    plot_confusion_matrix,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_length_distribution,
    plot_top_features,
)
from src.data_loader import load_raw_dataset
from src.preprocessing import preprocess_dataframe
from src.features import build_tfidf_features, transform_texts
from src.models import (
    train_test_split_data,
    train_svm,
    train_sgd,
    tune_linear_svc,
    tune_sgd_classifier,
)
from src.evaluation import summarize_model_results


def save_artifacts(artifacts_dir: Path, word_vec, char_vec, model) -> None:
    """
    Persist vectorizers and the primary classifier for later inference.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    dump(word_vec, artifacts_dir / "tfidf_word.joblib")
    dump(char_vec, artifacts_dir / "tfidf_char.joblib")
    dump(model, artifacts_dir / "svm_model.joblib")


def main():
    config = load_config()
    results_dir = Path(config["paths"]["results_dir"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    assets_dir = Path(config["paths"]["assets_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    # --- Load raw data ---
    df_raw = load_raw_dataset(config["paths"]["raw_data"])

    # Quick dataset shape
    print("Raw dataset shape:", df_raw.shape)

    # Visualize original rating distribution
    plot_rating_distribution(
        df_raw[config["data"]["rating_column"]],
        save_path=assets_dir / "rating_histogram.png",
    )

    # --- Preprocess ---
    X_text, y = preprocess_dataframe(
        df_raw,
        text_col=config["data"]["text_column"],
        rating_col=config["data"]["rating_column"],
        did_purchase_col=config["data"]["did_purchase_column"],
        config=config,
    )

    # Visualizations: sentiment distribution & wordcloud & lengths
    plot_class_distribution(y, save_path=assets_dir / "class_distribution.png")
    plot_wordcloud(
        X_text,
        "WordCloud of Cleaned Reviews",
        save_path=assets_dir / "wordcloud_all.png",
    )
    plot_length_distribution(
        X_text,
        save_path=assets_dir / "length_distribution.png",
    )

    # Save processed data
    processed_path = Path(config["paths"]["processed_data"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"text_clean": X_text, "sentiment": y}).to_parquet(processed_path, index=False)

    # --- Feature extraction (TF-IDF) ---
    X_all, word_vec, char_vec = build_tfidf_features(
        X_text,
        config["features"]["word_tfidf"],
        config["features"]["char_tfidf"],
    )

    # --- Train / Test split ---
    X_train, X_test, y_train, y_test = train_test_split_data(X_all, y, config["split"])
    _, text_test, _, _ = train_test_split_data(X_text, y, config["split"])

    # --- Validation split for light hyperparameter search ---
    X_train_inner, X_val, y_train_inner, y_val = train_test_split(
        X_train,
        y_train,
        test_size=config["split"].get("val_size", 0.1),
        random_state=config["split"]["random_state"],
        shuffle=True,
        stratify=y_train,
    )

    # --- Hyperparameter tuning ---
    svm_base = config["models"]["svm"]
    sgd_base = config["models"]["sgd"]
    svm_grid = svm_base.get("grid_C", [1.0])
    sgd_grid = sgd_base.get("grid_alpha", [1e-4])

    svm_tuned, best_c = tune_linear_svc(
        X_train_inner, y_train_inner, X_val, y_val, svm_base, svm_grid
    )
    sgd_tuned, best_alpha = tune_sgd_classifier(
        X_train_inner, y_train_inner, X_val, y_val, sgd_base, sgd_grid
    )

    print(f"Best LinearSVC C: {best_c}")
    print(f"Best SGD alpha: {best_alpha}")

    # --- Retrain on full training set with best params ---
    svm_model = train_svm(
        X_train, y_train, {**svm_base, "C": best_c} if best_c else svm_base
    )
    sgd_model = train_sgd(
        X_train, y_train, {**sgd_base, "alpha": best_alpha} if best_alpha else sgd_base
    )

    # --- Evaluate ---
    # SVM
    y_train_pred_svm = svm_model.predict(X_train)
    y_test_pred_svm = svm_model.predict(X_test)
    print("SVM Test Classification Report")
    print(classification_report(y_test, y_test_pred_svm))

    plot_confusion_matrix(
        y_test,
        y_test_pred_svm,
        title="Normalized Confusion Matrix – Linear SVM",
        save_path=assets_dir / "confusion_matrix_svm.png",
    )

    # SGD
    y_train_pred_sgd = sgd_model.predict(X_train)
    y_test_pred_sgd = sgd_model.predict(X_test)
    print("SGD Test Classification Report")
    print(classification_report(y_test, y_test_pred_sgd))

    plot_confusion_matrix(
        y_test,
        y_test_pred_sgd,
        title="Normalized Confusion Matrix – SGDClassifier",
        save_path=assets_dir / "confusion_matrix_sgd.png",
    )

    # --- Curves (decision_function for scores) ---
    class_order = list(svm_model.classes_)
    svm_scores = svm_model.decision_function(X_test)
    sgd_scores = sgd_model.decision_function(X_test)

    plot_precision_recall_curves(
        y_test, svm_scores, class_order, save_path=assets_dir / "pr_curves_svm.png"
    )
    plot_precision_recall_curves(
        y_test, sgd_scores, class_order, save_path=assets_dir / "pr_curves_sgd.png"
    )
    plot_roc_curves(
        y_test, svm_scores, class_order, save_path=assets_dir / "roc_curves_svm.png"
    )
    plot_roc_curves(
        y_test, sgd_scores, class_order, save_path=assets_dir / "roc_curves_sgd.png"
    )

    # --- Top features per class for SVM ---
    feature_names = np.hstack(
        [
            np.array(char_vec.get_feature_names_out()),
            np.array(word_vec.get_feature_names_out()),
        ]
    )
    plot_top_features(
        svm_model.coef_,
        feature_names,
        class_order,
        top_k=15,
        save_path=assets_dir / "top_features_svm.png",
    )

    # --- Sample misclassifications for quick error analysis ---
    misclassified = pd.DataFrame(
        {
            "text": text_test.reset_index(drop=True),
            "true": pd.Series(y_test).reset_index(drop=True),
            "pred": pd.Series(y_test_pred_svm),
        }
    )
    misclassified = misclassified[misclassified["true"] != misclassified["pred"]].head(30)
    misclassified.to_csv(results_dir / "sample_misclassified_svm.csv", index=False)

    # --- Summary tables (use numbers from your current run) ---
    svm_summary = summarize_model_results(
        "Linear SVM",
        y_train,
        y_train_pred_svm,
        y_test,
        y_test_pred_svm,
    )
    sgd_summary = summarize_model_results(
        "SGDClassifier",
        y_train,
        y_train_pred_sgd,
        y_test,
        y_test_pred_sgd,
    )

    results_df = pd.concat([svm_summary, sgd_summary], ignore_index=True)
    results_df["best_param"] = ["C=" + str(best_c), "alpha=" + str(best_alpha)]
    print(results_df)

    results_df.to_csv(config["paths"]["results_csv"], index=False)
    # Keep legacy path for convenience with the notebook
    results_df.to_csv("results_summary.csv", index=False)

    # --- Save artifacts for re-usable inference ---
    save_artifacts(artifacts_dir, word_vec, char_vec, svm_model)

    # --- Quick inference on unseen text (console + saved to disk) ---
    demo_reviews = [
        "Absolutely love this laptop, fast delivery and great build quality.",
        "Battery died in two days and support never replied.",
        "Packaging was fine but the product feels cheap.",
    ]
    demo_features = transform_texts(demo_reviews, word_vec, char_vec)
    demo_preds = svm_model.predict(demo_features)
    print("\nInference demo (Linear SVM, unseen text):")
    for review, label in zip(demo_reviews, demo_preds):
        print(f"{label}: {review}")

    pd.DataFrame({"text": demo_reviews, "predicted_sentiment": demo_preds}).to_csv(
        results_dir / "demo_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
