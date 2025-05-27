FROM apache/airflow:2.8.0-python3.11

# Switch to root to install system dependencies
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy project files
COPY --chown=airflow:root . /opt/airflow/workspace/

# Set working directory
WORKDIR /opt/airflow/workspace

# Install Python dependencies using pip from pyproject.toml
RUN pip install --no-cache-dir -e .

# Install additional Airflow providers
RUN pip install --no-cache-dir \
    apache-airflow-providers-docker \
    apache-airflow-providers-postgres

# Set environment variables
ENV AIRFLOW__CORE__DAGS_FOLDER=/opt/airflow/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor

# Copy DAGs
COPY --chown=airflow:root airflow/dags/ /opt/airflow/dags/

# Set the working directory back to airflow default
WORKDIR /opt/airflow
