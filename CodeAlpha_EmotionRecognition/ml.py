"""Classical models, metrics, confidence intervals and markdown helpers (framework-free, tested)."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def classical_models(seed=42):
    """MFCC-statistics + prosody (332 numbers per clip) par baseline models."""
    sc = lambda clf: Pipeline([("scale", StandardScaler()), ("clf", clf)])
    return {
        "Logistic Regression": sc(LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")),
        # SVC(probability=True) sklearn 1.9+ mein deprecated hai, isliye calibrated wrapper
        "SVM (RBF)": sc(CalibratedClassifierCV(SVC(kernel="rbf", C=10, class_weight="balanced"), cv=3, method="sigmoid")),
        "Random Forest": sc(RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced",
                                                   random_state=seed, n_jobs=-1)),
    }


def full_proba(model, X, n_classes):
    """predict_proba ko hamesha (N, n_classes) shape mein do, chahe training mein koi class na ho."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    out[:, np.asarray(model.classes_, int)] = p
    return out


def evaluate(y, pred):
    return {"accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0))}


def per_class_f1(y, pred, n_classes):
    return f1_score(y, pred, labels=list(range(n_classes)), average=None, zero_division=0)


def class_weights(y):
    y = np.asarray(y)
    n, k = len(y), len(np.unique(y))
    return {int(c): float(n / (k * (y == c).sum())) for c in np.unique(y)}


def bootstrap_ci(y, pred, metric, n=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a metric(y_true, y_pred). Returns (estimate, low, high)."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    vals = [metric(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, pred)), float(lo), float(hi)


def macro_f1(a, b):
    return f1_score(a, b, average="macro", zero_division=0)


def md_table(df: pd.DataFrame, index_name: str = "Model") -> str:
    cols = [index_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + [":---:"] * (len(cols) - 1)) + "|"]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in r.values]) + " |")
    return "\n".join(lines)
