from datetime import datetime, timedelta

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from airflow import DAG

# Default arguments for the DAG
default_args = {
    "owner": "hw-mlops-itmo-2025",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    "ml_pipeline_with_dvc",
    default_args=default_args,
    description="HW_MLOps_ITMO_2025: интеграция Airflow <-> DVC",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["ml", "pipeline", "catboost", "dvc"],
)

dvc_pull_data = BashOperator(
    task_id="dvc_pull_data",
    bash_command="""
        cd /opt/airflow/workspace && \
        dvc pull
    """,
    env={},
    dag=dag,
)


def check_data_availability():
    """Check if required data directories exist"""
    from pathlib import Path

    data_dir = Path("/opt/airflow/workspace/data/embeddings")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    datasets = ["enhancers", "promoter_all", "splice_sites_all", "H3K9me3", "H4K20me1"]
    for dataset in datasets:
        dataset_path = data_dir / dataset
        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset} not found at {dataset_path}")

    print("Data availability check completed")


check_data = PythonOperator(
    task_id="check_data_availability",
    python_callable=check_data_availability,
    dag=dag,
)

process_data = BashOperator(
    task_id="process_data",
    bash_command="cd /opt/airflow/workspace && python src/process.py",
    dag=dag,
)

train_models = BashOperator(
    task_id="train_models",
    bash_command="cd /opt/airflow/workspace && python src/train.py",
    dag=dag,
)

evaluate_models = BashOperator(
    task_id="evaluate_models",
    bash_command="cd /opt/airflow/workspace && python src/evaluate.py",
    dag=dag,
)

dvc_add_models = BashOperator(
    task_id="dvc_add_models",
    bash_command="""
        cd /opt/airflow/workspace && \
        dvc add models && \
        echo "Models added to DVC"
    """,
    dag=dag,
)

dvc_add_metrics = BashOperator(
    task_id="dvc_add_metrics",
    bash_command="""
        cd /opt/airflow/workspace && \
        dvc add metrics/enhancers.json && \
        dvc add metrics/promoter_all.json && \
        dvc add metrics/splice_sites_all.json && \
        dvc add metrics/H3K9me3.json && \
        dvc add metrics/H4K20me1.json && \
        echo "Metrics added to DVC"
    """,
    dag=dag,
)

dvc_push = BashOperator(
    task_id="dvc_push",
    bash_command="""
        cd /opt/airflow/workspace && \
        dvc push && \
        echo "DVC push completed"
    """,
    env={},
    dag=dag,
)

dvc_pull_data >> check_data >> process_data >> train_models >> evaluate_models
evaluate_models >> dvc_add_models >> dvc_add_metrics >> dvc_push
