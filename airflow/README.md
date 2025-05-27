TODO: MOVE THIS TO PROJECT'S README.md

# Airflow ML Pipeline

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM and 2 CPUs available for Docker

### Quick Start

1. **Set environment variables** (copy `airflow.env` to `.env` or export manually):
   ```bash
   export AIRFLOW_UID=50000
   export AIRFLOW_PROJ_DIR=.
   export _AIRFLOW_WWW_USER_USERNAME=airflow
   export _AIRFLOW_WWW_USER_PASSWORD=airflow
   ```

2. **Initialize and start Airflow**:
   ```bash
   # Initialize the database and create admin user
   docker-compose -f docker-compose-airflow.yaml up airflow-init
   
   # Start all services
   docker-compose -f docker-compose-airflow.yaml up -d
   ```

3. **Access Airflow Web UI**:
   - URL: http://localhost:8080
   - Username: `airflow`
   - Password: `airflow`

4. **Stop services**:
   ```bash
   docker-compose -f docker-compose-airflow.yaml down
   ```

### DAG Overview

The `ml_pipeline` DAG includes the following tasks:

1. **check_data_availability**: Validates that required data directories exist
2. **process_data**: Runs `src/process.py` to add L2 norm features
3. **train_models**: Runs `src/train.py` to train CatBoost models
4. **evaluate_models**: Runs `src/evaluate.py` to evaluate model performance

### Triggering DAG
1. List available DAGs
```bash
docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags list
```
2. Unpause `ml_pipeline`
```
docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags unpause ml_pipeline
```
3. Trigger `ml_pipeline`
```bash
docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags trigger ml_pipeline
```
4. Show state:
```bash
docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags state ml_pipeline <dag_run_id
```

---
### Reference list

1. **List all DAGs:**
   ```bash
   docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags list
   ```

2. **Enable (unpause) a DAG:**
   ```bash
   docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags unpause ml_pipeline
   ```

3. **Trigger a manual DAG run:**
   ```bash
   docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags trigger ml_pipeline
   ```

4. **Check DAG run status:**
   ```bash
   docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags state ml_pipeline <execution_date>
   ```

5. **Check individual task statuses:**
   ```bash
   docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow tasks states-for-dag-run ml_pipeline <execution_date>
   ```

### 🔧 **Additional Useful Commands**

- **Pause a DAG:**
  ```bash
  docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags pause ml_pipeline
  ```

- **List DAG runs:**
  ```bash
  docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow dags list-runs ml_pipeline
  ```

- **Test a specific task:**
  ```bash
  docker compose -f docker-compose-airflow.yaml exec airflow-webserver python -m airflow tasks test ml_pipeline <task_id> <execution_date>
  ```