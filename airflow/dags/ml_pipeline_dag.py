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
    "ml_pipeline",
    default_args=default_args,
    description="HW_MLOps_ITMO_2025",
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    tags=["ml", "pipeline", "catboost"],
)

# Task 1: Data Processing
process_data = BashOperator(
    task_id="process_data",
    bash_command="cd /opt/airflow/workspace && python src/process.py",
    dag=dag,
)

# Task 2: Model Training
train_models = BashOperator(
    task_id="train_models",
    bash_command="cd /opt/airflow/workspace && python src/train.py",
    dag=dag,
)

# Task 3: Model Evaluation
evaluate_models = BashOperator(
    task_id="evaluate_models",
    bash_command="cd /opt/airflow/workspace && python src/evaluate.py",
    dag=dag,
)


# Add a task to check data availability
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

check_data >> process_data >> train_models >> evaluate_models
