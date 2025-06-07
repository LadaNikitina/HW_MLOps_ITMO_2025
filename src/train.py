import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

DATASETS = [
    "enhancers",
    "promoter_all",
    "splice_sites_all",
    "H3K9me3",
    "H4K20me1",
]

DATA_DIR = Path("data/processed_embeddings")
MODEL_DIR = Path("models")


def load_data(dataset_path: Path):
    df_train = pd.read_csv(dataset_path / "train.csv")
    df_valid = pd.read_csv(dataset_path / "val.csv")
    return df_train, df_valid


def create_classifier(model_name, **kwargs):
    if model_name == "catboost":
        return CatBoostClassifier(verbose=50, task_type="CPU", **kwargs)
    elif model_name == "lightgbm":
        if "depth" in kwargs:
            kwargs["max_depth"] = kwargs.pop("depth")
        if "iterations" in kwargs:
            kwargs["n_estimators"] = kwargs.pop("iterations")
        if "lr" in kwargs:
            kwargs["learning_rate"] = kwargs.pop("lr")

        kwargs.setdefault("num_leaves", 31)
        kwargs.setdefault("min_split_gain", 0.0)
        kwargs.setdefault("min_child_samples", 20)

        return LGBMClassifier(verbose=300, eval_metric="logloss", **kwargs)
    elif model_name == "random_forest":
        filtered = {"max_depth": kwargs.get("depth"), "n_estimators": kwargs.get("iterations", 100)}
        return RandomForestClassifier(**{k: v for k, v in filtered.items() if v is not None})
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_classifier(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf


def save_model(clf, path: Path, model_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "catboost":
        clf.save_model(path.with_suffix(".cbm"))
    else:
        joblib.dump(clf, path.with_suffix(".pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["catboost", "lightgbm", "random_forest"], default="catboost"
    )
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--out_suffix", type=str, required=True)
    args = parser.parse_args()

    for dataset in DATASETS:
        print(f"\nTraining on: {dataset}")
        dataset_path = DATA_DIR / dataset
        df_train, df_valid = load_data(dataset_path)

        X_train = df_train.drop(columns=["label"])
        y_train = df_train["label"]
        X_valid = df_valid.drop(columns=["label"])
        y_valid = df_valid["label"]

        with mlflow.start_run(run_name=f"{dataset}_{args.model}"):
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("model", args.model)
            mlflow.log_param("depth", args.depth)
            mlflow.log_param("learning_rate", args.lr)
            mlflow.log_param("iterations", args.iterations)

            clf = create_classifier(
                args.model,
                depth=args.depth,
                learning_rate=args.lr,
                n_estimators=args.iterations,
            )

            clf = train_classifier(clf, X_train, y_train)

            model_out_path = MODEL_DIR / args.out_suffix / dataset / f"{args.model}_model"
            save_model(clf, model_out_path, args.model)

            mlflow.log_artifact(
                str(model_out_path.with_suffix(".cbm" if args.model == "catboost" else ".pkl")),
                artifact_path="models",
            )

            print(f"Model saved to {model_out_path}")
