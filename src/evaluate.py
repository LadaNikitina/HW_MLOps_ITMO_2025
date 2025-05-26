import json
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

DATASETS = [
    "enhancers",
    "promoter_all",
    "splice_sites_all",
    "H3K9me3",
    "H4K20me1",
]

METRICS = {
    "enhancers": "MCC",
    "promoter_all": "F1",
    "splice_sites_all": "Accuracy",
    "H3K9me3": "MCC",
    "H4K20me1": "MCC",
}

PROCESSED_DIR = Path("data/processed_embeddings")
MODEL_DIR = Path("models")
METRICS_DIR = Path("metrics")

def load_data(dataset_path: Path):
    df_test = pd.read_csv(dataset_path / "test.csv")
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]
    return X_test, y_test

def load_model(model_path: Path):
    clf = CatBoostClassifier()
    clf.load_model(model_path)
    return clf

def evaluate_model(clf, X_test, y_test, metric):
    y_pred = clf.predict(X_test)
    if y_pred.ndim == 1:
        y_pred = (y_pred > 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred, axis=1)

    if metric == "F1":
        return f1_score(y_test, y_pred, average="binary")
    elif metric == "Accuracy":
        return accuracy_score(y_test, y_pred)
    else:
        return matthews_corrcoef(y_test, y_pred)

if __name__ == "__main__":
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        print(f"\nEvaluating {dataset}...")

        X_test, y_test = load_data(PROCESSED_DIR / dataset)
        model_path = MODEL_DIR / dataset / "model.cbm"
        clf = load_model(model_path)

        metric_name = METRICS[dataset]
        score = evaluate_model(clf, X_test, y_test, metric_name)

        result = {metric_name: round(score, 4)}
        out_path = METRICS_DIR / f"{dataset}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"{metric_name} = {score:.4f} → saved to {out_path}")
