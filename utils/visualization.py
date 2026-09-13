from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def plot_rating_distribution(series, save_path: Optional[str] = None):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=series)
    plt.xlabel("Original Rating (1–5)")
    plt.ylabel("Count")
    plt.title("Distribution of Review Ratings")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_class_distribution(series, save_path: Optional[str] = None):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=series, order=["Unhappy", "Ok", "Happy"])
    plt.xlabel("Sentiment Class")
    plt.ylabel("Count")
    plt.title("Sentiment Class Distribution")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_wordcloud(text_series, title: str, save_path: Optional[str] = None):
    text = " ".join(text_series.astype(str).tolist())
    stopwords = set(STOPWORDS)

    wc = WordCloud(
        background_color="white",
        stopwords=stopwords,
        max_words=250,
        max_font_size=40,
        width=800,
        height=400,
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Optional[str] = None):
    labels = ["Unhappy", "Ok", "Happy"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_precision_recall_curves(y_true, decision_scores, labels, save_path: Optional[str] = None):
    """
    Plot one-vs-rest precision-recall curves using decision scores.
    """
    y_bin = label_binarize(y_true, classes=labels)
    plt.figure(figsize=(6, 5))
    for idx, label in enumerate(labels):
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], decision_scores[:, idx])
        plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (one-vs-rest)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_curves(y_true, decision_scores, labels, save_path: Optional[str] = None):
    """
    Plot one-vs-rest ROC curves using decision scores.
    """
    y_bin = label_binarize(y_true, classes=labels)
    plt.figure(figsize=(6, 5))
    for idx, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], decision_scores[:, idx])
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curves (one-vs-rest)")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_length_distribution(text_series, save_path: Optional[str] = None):
    """
    Plot histogram of review lengths (in tokens).
    """
    lengths = text_series.astype(str).apply(lambda x: len(x.split()))
    plt.figure(figsize=(6, 4))
    sns.histplot(lengths, bins=40, kde=True)
    plt.xlabel("Review length (tokens)")
    plt.ylabel("Count")
    plt.title("Distribution of Cleaned Review Lengths")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_top_features(coef, feature_names, class_labels, top_k=15, save_path: Optional[str] = None):
    """
    Visualize top positive features per class from a linear model.
    """
    num_classes = len(class_labels)
    fig, axes = plt.subplots(1, num_classes, figsize=(5 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]

    for idx, label in enumerate(class_labels):
        # For multi-class LinearSVC, coef shape is (n_classes, n_features)
        weights = coef[idx]
        top_indices = np.argsort(weights)[-top_k:]
        top_features = [feature_names[i] for i in top_indices]
        top_weights = weights[top_indices]
        axes[idx].barh(top_features, top_weights)
        axes[idx].set_title(f"Top features: {label}")
        axes[idx].set_xlabel("Weight")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
