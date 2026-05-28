import sqlite3
import pandas as pd
from google.cloud import storage
import os
import io
from dotenv import load_dotenv

load_dotenv()

SQLITE_PATH   = os.getenv("SQLITE_PATH", "data/raw/travel.sqlite")
BRONZE_BUCKET = os.getenv("GCS_BUCKET_BRONZE", "travel-rag-bronze")

def get_sqlite_tables(sqlite_path: str) -> list[str]:
    conn   = sqlite3.connect(sqlite_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()
    return [t[0] for t in tables]

def upload_df_to_gcs(df: pd.DataFrame, bucket_name: str, blob_path: str):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_path)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")

def run_bronze_ingestion():
    tables = get_sqlite_tables(SQLITE_PATH)
    print(f"Found {len(tables)} tables: {tables}")

    for table in tables:
        conn = sqlite3.connect(SQLITE_PATH)
        df   = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()

        df["_ingested_at"] = pd.Timestamp.now().isoformat()
        df["_source"]      = "travel2.sqlite"

        date_str  = pd.Timestamp.now().strftime("%Y-%m-%d")
        blob_path = f"bronze/{table}/{date_str}.csv"
        upload_df_to_gcs(df, BRONZE_BUCKET, blob_path)
        print(f"  Uploaded {len(df)} rows → gs://{BRONZE_BUCKET}/{blob_path}")

if __name__ == "__main__":
    run_bronze_ingestion()