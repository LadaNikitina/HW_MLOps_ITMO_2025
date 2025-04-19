import os
import torch

from .dataset import load_nucleotide_transformer
from .embeddings import get_dnabert2_embeddings
from .model import train_classifier, evaluate_model
from transformers import AutoTokenizer, BertConfig, AutoModel

def classify_with_dnabert(dataset_name, metric, embedding_fn=get_dnabert2_embeddings):
    batch_size = 128
    train_loader, valid_loader, test_loader = load_nucleotide_transformer(batch_size, valid_split=0.1, dataset_name=dataset_name)

    model_name = "zhihan1996/DNABERT-2-117M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = BertConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
    model.eval()

    X_train, y_train = embedding_fn(train_loader, tokenizer, model)
    X_valid, y_valid = embedding_fn(valid_loader, tokenizer, model)
    X_test, y_test = embedding_fn(test_loader, tokenizer, model)

    clf = train_classifier(X_train, y_train, X_valid, y_valid, metric)
    metric_value = evaluate_model(clf, X_test, y_test, metric)
    
    print(f"{metric}: {metric_value:.4f}")
    return metric_value