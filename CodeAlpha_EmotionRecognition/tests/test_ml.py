import numpy as np

import ml


def test_full_proba_handles_missing_class():
    class M:
        classes_ = np.array([0, 2])
        predict_proba = staticmethod(lambda X: np.tile([0.3, 0.7], (len(X), 1)))
    p = ml.full_proba(M(), np.zeros((4, 3)), 4)
    assert p.shape == (4, 4) and np.allclose(p.sum(1), 1) and (p[:, [1, 3]] == 0).all()


def test_metrics_and_class_weights():
    y, pred = np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 1, 0, 2, 2])
    m = ml.evaluate(y, pred)
    assert abs(m["accuracy"] - 5 / 6) < 1e-9 and 0 < m["macro_f1"] <= 1
    w = ml.class_weights(np.array([0, 0, 0, 1]))
    assert w[1] > w[0] and abs(w[0] * 3 + w[1] - 4) < 1e-9          # rare class ko zyada weight
    assert len(ml.per_class_f1(y, pred, 4)) == 4


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    pred = np.where(rng.random(200) < 0.7, y, rng.integers(0, 3, 200))
    est, lo, hi = ml.bootstrap_ci(y, pred, ml.macro_f1, n=200)
    assert lo <= est <= hi and hi - lo > 0


def test_classical_models_are_warning_free_api():
    ms = ml.classical_models()
    assert set(ms) == {"Logistic Regression", "SVM (RBF)", "Random Forest"}
