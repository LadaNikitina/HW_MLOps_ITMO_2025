from pathlib import Path
import pandas as pd
import numpy as np

RAW_ROOT = Path("data/embeddings")
OUT_ROOT = Path("data/processed_embeddings")

DATASETS = [
    "enhancers",
    "promoter_all",
    "splice_sites_all",
    "H3K9me3",
    "H4K20me1",
]

SPLITS = ["train.csv", "val.csv", "test.csv"]

def add_l2_norm_feature(df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=["label"]).values
    norms = np.linalg.norm(features, axis=1)
    df = df.copy()
    df["l2_norm"] = norms
    return df

def process_dataset(dataset_name: str):
    in_dir = RAW_ROOT / dataset_name
    out_dir = OUT_ROOT / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        in_path = in_dir / split
        out_path = out_dir / split

        if not in_path.exists():
            print(f"Skip missing file: {in_path}")
            continue

        df = pd.read_csv(in_path)
        df = add_l2_norm_feature(df)
        df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    for ds in DATASETS:
        process_dataset(ds)
