import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

API_ENDPOINTS = {
    "catboost": "http://localhost:8001/predict?model_version=catboost",
    "lgbm": "http://localhost:8002/predict?model_version=lgbm",
    "rf": "http://localhost:8003/predict?model_version=rf",
}

DATASETS = ["enhancers", "promoter_all", "splice_sites_all", "H3K9me3", "H4K20me1"]

METRICS_MAPPING = {
    "enhancers": "MCC",
    "promoter_all": "F1",
    "splice_sites_all": "Accuracy",
    "H3K9me3": "MCC",
    "H4K20me1": "MCC",
}


def load_test_data(dataset: str) -> Tuple[List[List[float]], List[int]]:
    test_file = Path(f"data/processed_embeddings/{dataset}/test.csv")

    if not test_file.exists():
        print(f"Warning: {test_file} not found, skipping {dataset}")
        return [], []

    df = pd.read_csv(test_file)

    feature_columns = [col for col in df.columns if col != "label" and col != "l2_norm"]
    X = df[feature_columns].values.tolist()
    y = df["label"].values.tolist()

    return X, y


def make_prediction_request(
    features: List[float], dataset: str, model_endpoint: str
) -> Tuple[float, float]:
    request_data = {"features": features, "dataset": dataset}

    start_time = time.time()
    try:
        response = requests.post(
            model_endpoint,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"][0]
            response_time = end_time - start_time
            return prediction, response_time
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None, None

    except Exception as e:
        print(f"Request failed: {e}")
        return None, None


def benchmark_model(
    model_name: str, endpoint: str, dataset: str, X_test: List[List[float]], y_test: List[int]
) -> Dict:
    print(f"Benchmarking {model_name} on {dataset}...")

    predictions = []
    response_times = []
    failed_requests = 0

    for i, (features, true_label) in enumerate(zip(X_test, y_test)):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(X_test)}")

        pred, resp_time = make_prediction_request(features, dataset, endpoint)

        if pred is not None and resp_time is not None:
            predictions.append(pred)
            response_times.append(resp_time)
        else:
            failed_requests += 1

    if len(predictions) == 0:
        print(f"No successful predictions for {model_name} on {dataset}")
        return None

    y_true = y_test[: len(predictions)]
    y_pred = [1 if p > 0.5 else 0 for p in predictions]

    unique_labels = set(y_true)
    is_binary = len(unique_labels) <= 2

    metrics = {
        "model": model_name,
        "dataset": dataset,
        "samples_tested": len(predictions),
        "failed_requests": failed_requests,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average="binary" if is_binary else "macro", zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, average="binary" if is_binary else "macro", zero_division=0
        ),
        "f1": f1_score(y_true, y_pred, average="binary" if is_binary else "macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "avg_response_time": statistics.mean(response_times),
        "min_response_time": min(response_times),
        "max_response_time": max(response_times),
        "median_response_time": statistics.median(response_times),
        "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0,
        "requests_per_second": len(predictions) / sum(response_times)
        if sum(response_times) > 0
        else 0,
    }

    return metrics


def run_full_benchmark() -> List[Dict]:
    results = []

    for dataset in DATASETS:
        print(f"\n=== Testing dataset: {dataset} ===")

        X_test, y_test = load_test_data(dataset)

        if len(X_test) == 0:
            print(f"Skipping {dataset} - no data loaded")
            continue

        print(f"Loaded {len(X_test)} samples")

        for model_name, endpoint in API_ENDPOINTS.items():
            result = benchmark_model(model_name, endpoint, dataset, X_test, y_test)
            if result:
                results.append(result)

    return results


def save_detailed_results(results: List[Dict], filename: str = "benchmark_detailed_results.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {filename}")


if __name__ == "__main__":
    print("Starting API benchmark...")

    results = run_full_benchmark()

    if results:
        save_detailed_results(results)
