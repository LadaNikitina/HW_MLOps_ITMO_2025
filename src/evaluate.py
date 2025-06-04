import json
import joblib
from pathlib import Path
import mlflow

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
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

EXPERIMENTS = {
    "catboost": "catboost",
    "rf": "random_forest",
    "lgbm": "lightgbm",
}

PROCESSED_DIR = Path("data/processed_embeddings")
MODEL_DIR = Path("models")
METRICS_DIR = Path("mlflow_metrics")


def load_data(dataset_path: Path):
    df_test = pd.read_csv(dataset_path / "test.csv")
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]
    return X_test, y_test


def load_model(path: Path, model_type: str):
    if model_type == "catboost":
        model = CatBoostClassifier()
        model.load_model(str(path.with_suffix(".cbm")))
        return model
    else:
        return joblib.load(path.with_suffix(".pkl"))


def evaluate_model(clf, X_test, y_test, metric):
    y_pred = clf.predict(X_test)

    if hasattr(y_pred, "ndim") and y_pred.ndim == 1:
        y_pred = (y_pred > 0.5).astype(int) if len(np.unique(y_test)) == 2 else y_pred
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

    for exp_name, model_name in EXPERIMENTS.items():
        mlflow.set_experiment(f"eval_{exp_name}")

        for dataset in DATASETS:
            model_path = MODEL_DIR / exp_name / dataset / f"{model_name}_model"
            if not model_path.with_suffix(".cbm").exists() and not model_path.with_suffix(".pkl").exists():
                print(f"Skipping missing model: {model_path}")
                continue

            print(f"\nEvaluating {dataset} with {model_name}...")

            X_test, y_test = load_data(PROCESSED_DIR / dataset)
            clf = load_model(model_path, model_name)

            metric_name = METRICS[dataset]
            score = evaluate_model(clf, X_test, y_test, metric_name)

            result = {metric_name: round(score, 4)}
            out_path = METRICS_DIR / f"{dataset}_{exp_name}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            with mlflow.start_run(run_name=f"{dataset}_{exp_name}"):
                mlflow.log_param("dataset", dataset)
                mlflow.log_param("model", model_name)
                mlflow.log_param("metric", metric_name)
                mlflow.log_metric(metric_name, score)
                mlflow.log_artifact(str(out_path), artifact_path="mlflow_metrics")

            print(f"{metric_name} = {score:.4f} → saved to {out_path}")