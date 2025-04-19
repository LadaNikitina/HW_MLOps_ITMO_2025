import os
from src.train import classify_with_dnabert

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    DATASETS = [
        ("promoter_all", "F1"),
        ("enhancers", "MCC"),
        ("splice_sites_all", "Accuracy"),
        ("H3", "MCC"),
        ("H4", "MCC")
    ]
    
    result_metrics = {}
    
    for dataset, metric in DATASETS:
        result_metrics[dataset] = classify_with_dnabert(dataset, metric)
    
    max_name_len = max(len(name) for name in result_metrics.keys())
    line_format = f"{{:<{max_name_len}}} : {{:.4f}}"
    
    print("Metrics per dataset:\n")
    for dataset, score in result_metrics.items():
        print(line_format.format(dataset, score))
    
    mean_score = sum(result_metrics.values()) / len(result_metrics)
    print("\n" + "-" * (max_name_len + 12))
    print(f"{'Average'.ljust(max_name_len)} : {mean_score:.4f}")

if __name__ == "__main__":
    main()
