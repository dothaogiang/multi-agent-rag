from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

# Để Airflow tìm được các module trong dự án
sys.path.insert(0, "/workspaces/multi-agent-rag")

from data_pipeline.processors.bronze_ingestion import run_bronze_ingestion
from data_pipeline.processors.silver_transform import run_silver_transform

default_args = {
    "owner":            "travel-rag",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="travel_etl_pipeline",
    description="SQLite → GCS Bronze → GCS Silver + PostgreSQL Silver",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["travel", "etl", "gcs", "rag"],
) as dag:

    task_bronze = PythonOperator(
        task_id="bronze_ingest",
        python_callable=run_bronze_ingestion,
        doc_md="Read SQLite → upload raw CSVs to GCS bronze/",
    )

    task_silver = PythonOperator(
        task_id="silver_transform",
        python_callable=run_silver_transform,
        doc_md="Read GCS bronze → clean → GCS silver + PostgreSQL silver",
    )

    task_bronze >> task_silver