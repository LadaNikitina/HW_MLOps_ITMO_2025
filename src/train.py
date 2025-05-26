from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier

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

def create_classifier():
    return CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=4,
        verbose=50,
        task_type="GPU",
    )

def train_classifier(clf, X_train, y_train, X_valid, y_valid):
    clf.fit(X_train, y_train)
    return clf

def save_model(clf, path: Path):
    clf.save_model(path)

if __name__ == "__main__":
    for dataset in DATASETS:
        print(f"\nTraining on: {dataset}")
        dataset_path = DATA_DIR / dataset
        df_train, df_valid = load_data(dataset_path)

        X_train = df_train.drop(columns=["label"])
        y_train = df_train["label"]
        X_valid = df_valid.drop(columns=["label"])
        y_valid = df_valid["label"]

        clf = create_classifier()
        clf = train_classifier(clf, X_train, y_train, X_valid, y_valid)

        model_out_path = MODEL_DIR / dataset / "model.cbm"
        model_out_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(clf, model_out_path)
        print(f"Model saved to {model_out_path}")
