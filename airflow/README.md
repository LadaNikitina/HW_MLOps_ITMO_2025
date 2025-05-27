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