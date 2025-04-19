from catboost import CatBoostClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

def train_classifier(X_train, y_train, X_valid, y_valid, metric):
    eval_metric = "F1" if metric == "MCC" else metric
        
    clf = CatBoostClassifier(
        iterations=3_000,
        learning_rate=0.02,
        depth=4,
        task_type="GPU",
        eval_metric=eval_metric,
        early_stopping_rounds=100,
        use_best_model=True,
        verbose=50
    )

    clf.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    return clf

def evaluate_model(clf, X_test, y_test, metric):
    y_pred = clf.predict(X_test)

    if metric == "F1":
        return f1_score(y_test, y_pred, average="binary")
    elif metric == "Accuracy":
        return accuracy_score(y_test, y_pred)
    else:
        return matthews_corrcoef(y_test, y_pred)