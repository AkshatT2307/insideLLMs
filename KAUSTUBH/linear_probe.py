"""
linear_probe.py — Train and evaluate a linear classifier on activations.

Given activations from a single layer and component, trains logistic
regression to predict domain label. Returns test accuracy.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def probe_accuracy(X: np.ndarray, y: np.ndarray,
                   test_size: float = 0.3, seed: int = 42) -> dict:
    """
    Train a linear probe and return test accuracy.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Activation vectors.
    y : np.ndarray, shape (N,)
        Integer domain labels.
    test_size : float
        Fraction of data for testing.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        'accuracy' : float
        'per_class_accuracy' : dict[int, float]
        'n_train' : int
        'n_test' : int
    """
    # Single stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                     random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        C=1.0,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))

    # Per-class accuracy
    classes = np.unique(y)
    per_class = {}
    for c in classes:
        mask = y_test == c
        if mask.sum() > 0:
            per_class[int(c)] = float(np.mean(y_pred[mask] == y_test[mask]))

    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
