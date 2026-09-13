from typing import Tuple
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)


def train_test_split_data(X, y, split_cfg: dict):
    test_size = split_cfg.get("test_size", 0.3)
    random_state = split_cfg.get("random_state", 101)
    stratify = y if split_cfg.get("stratify", True) else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )


def train_svm(X_train, y_train, model_cfg: dict) -> LinearSVC:
    model = LinearSVC(
        class_weight=model_cfg.get("class_weight", "balanced"),
        C=model_cfg.get("C", 1.0),
        loss="hinge",
    )
    model.fit(X_train, y_train)
    return model


def train_sgd(X_train, y_train, model_cfg: dict) -> SGDClassifier:
    model = SGDClassifier(
        class_weight=model_cfg.get("class_weight", "balanced"),
        max_iter=model_cfg.get("max_iter", 300),
        tol=model_cfg.get("tol", 1e-3),
        loss=model_cfg.get("loss", "hinge"),
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_macro_metrics(y_true, y_pred) -> Tuple[float, float, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    acc = accuracy_score(y_true, y_pred)
    return acc, precision, recall, f1


def tune_linear_svc(
    X_train, y_train, X_val, y_val, base_cfg: dict, c_grid
) -> Tuple[LinearSVC, float]:
    """
    Simple grid search over C for LinearSVC using macro F1 on validation data.
    """
    best_f1 = -1.0
    best_model = None
    best_c = None
    for c in c_grid:
        model = LinearSVC(
            class_weight=base_cfg.get("class_weight", "balanced"),
            C=c,
            loss="hinge",
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_model = model
            best_c = c
    return best_model, best_c


def tune_sgd_classifier(
    X_train, y_train, X_val, y_val, base_cfg: dict, alpha_grid
) -> Tuple[SGDClassifier, float]:
    """
    Simple grid search over alpha for SGDClassifier using macro F1 on validation data.
    """
    best_f1 = -1.0
    best_model = None
    best_alpha = None
    for alpha in alpha_grid:
        model = SGDClassifier(
            class_weight=base_cfg.get("class_weight", "balanced"),
            max_iter=base_cfg.get("max_iter", 300),
            tol=base_cfg.get("tol", 1e-3),
            loss=base_cfg.get("loss", "hinge"),
            alpha=alpha,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, average="macro")
        if score > best_f1:
            best_f1 = score
            best_model = model
            best_alpha = alpha
    return best_model, best_alpha
