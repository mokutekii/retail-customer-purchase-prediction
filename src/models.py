"""Model factory/wrappers for LR, DT, RF, (optional) GB, and MLP."""
from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def make_model(name: str, **kwargs) -> Any:
    name = name.lower()
    if name in ["lr", "logreg", "logistic"]:
        # class_weight='balanced' helps with imbalance
        return LogisticRegression(max_iter=1000, class_weight='balanced', **kwargs)
    if name in ["dt", "tree"]:
        return DecisionTreeClassifier(**kwargs)
    if name in ["rf", "random_forest"]:
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, **kwargs)
    if name in ["gb", "gboost", "gbdt"]:
        return GradientBoostingClassifier(**kwargs)
    raise ValueError(f"Unknown model name: {name}")
