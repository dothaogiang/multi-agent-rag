import pandas as pd
from google.cloud import storage
from sqlalchemy import create_engine
import os
import io
from dotenv import load_dotenv

load_dotenv()

BRONZE_BUCKET = os.getenv("GCS_BUCKET_BRONZE", "travel-rag-bronze")
SILVER_BUCKET = os.getenv("GCS_BUCKET_SILVER",  "travel-rag-silver")
POSTGRES_URL  = os.getenv("POSTGRES_URL",
    "postgresql://travel_user:travel_pass@localhost:5432/travel_db")

def read_latest_csv_from_gcs(bucket_name: str, table: str) -> pd.DataFrame:
    client  = storage.Client()
    prefix  = f"bronze/{table}/"
    blobs   = sorted(
        client.list_blobs(bucket_name, prefix=prefix),
        key=lambda b: b.updated, reverse=True
    )
    if not blobs:
        raise FileNotFoundError(
            f"No files found at gs://{bucket_name}/{prefix}"
        )
    content = blobs[0].download_as_text()
    return pd.read_csv(io.StringIO(content))

def upload_to_gcs_silver(df: pd.DataFrame, table: str):
    client    = storage.Client()
    bucket    = client.bucket(SILVER_BUCKET)
    date_str  = pd.Timestamp.now().strftime("%Y-%m-%d")
    blob_path = f"silver/{table}/{date_str}.csv"
    blob      = bucket.blob(blob_path)
    buffer    = io.StringIO()
    df.to_csv(buffer, index=False)
    blob.upload_from_string(buffer.getvalue(), content_type="text/csv")
    print(f"  Uploaded → gs://{SILVER_BUCKET}/{blob_path}")

# --- cleaner functions ---

def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["flight_id"]).copy()

    df["flight_id"] = df["flight_id"].astype(int)

    df["departure_airport"] = (
        df["departure_airport"]
        .str.upper()
        .str.strip()
    )

    df["arrival_airport"] = (
        df["arrival_airport"]
        .str.upper()
        .str.strip()
    )

    df["status"] = (
        df["status"]
        .str.upper()
        .str.strip()
    )

    df["scheduled_departure"] = pd.to_datetime(
        df["scheduled_departure"],
        errors="coerce"
    )

    df["scheduled_arrival"] = pd.to_datetime(
        df["scheduled_arrival"],
        errors="coerce"
    )

    df["actual_departure"] = pd.to_datetime(
        df["actual_departure"],
        errors="coerce"
    )

    df["actual_arrival"] = pd.to_datetime(
        df["actual_arrival"],
        errors="coerce"
    )

    df["_transformed_at"] = pd.Timestamp.now().isoformat()

    return df

def clean_hotels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["id"]).copy()

    df["id"] = df["id"].astype(int)

    df["name"] = (
        df["name"]
        .str.strip()
        .str.title()
    )

    df["location"] = (
        df["location"]
        .str.strip()
    )

    df["price_tier"] = (
        df["price_tier"]
        .str.lower()
        .str.strip()
    )

    df["checkin_date"] = pd.to_datetime(
        df["checkin_date"],
        errors="coerce"
    )

    df["checkout_date"] = pd.to_datetime(
        df["checkout_date"],
        errors="coerce"
    )

    df["booked"] = df["booked"].astype(bool)

    df["_transformed_at"] = pd.Timestamp.now().isoformat()

    return df

def clean_car_rentals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["id"]).copy()

    df["id"] = df["id"].astype(int)

    df["name"] = (
        df["name"]
        .str.strip()
        .str.title()
    )

    df["location"] = (
        df["location"]
        .str.strip()
    )

    df["price_tier"] = (
        df["price_tier"]
        .str.lower()
        .str.strip()
    )

    df["start_date"] = pd.to_datetime(
        df["start_date"],
        errors="coerce"
    )

    df["end_date"] = pd.to_datetime(
        df["end_date"],
        errors="coerce"
    )

    df["booked"] = df["booked"].astype(bool)

    df["_transformed_at"] = pd.Timestamp.now().isoformat()

    return df

def clean_trip_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["id"]).copy()
    df["id"]              = df["id"].astype(int)
    df["name"]            = df["name"].str.strip().str.title()
    df["location"]        = df["location"].str.strip()
    df["keywords"]        = df["keywords"].str.lower().str.strip()
    df["details"]         = df["details"].str.strip()
    df["booked"]          = df["booked"].astype(bool)
    df["_transformed_at"] = pd.Timestamp.now().isoformat()
    return df

CLEANERS = {
    "flights":              clean_flights,
    "hotels":               clean_hotels,
    "car_rentals":          clean_car_rentals,
    "trip_recommendations": clean_trip_recommendations,
}

def run_silver_transform():
    engine = create_engine(POSTGRES_URL)

    for table, cleaner_fn in CLEANERS.items():
        print(f"\nProcessing: {table}")
        df_raw   = read_latest_csv_from_gcs(BRONZE_BUCKET, table)
        print(f"  Read {len(df_raw)} rows from GCS bronze")

        df_clean = cleaner_fn(df_raw)
        print(f"  Cleaned → {len(df_clean)} rows")

        upload_to_gcs_silver(df_clean, table)

        df_clean.to_sql(table, engine, schema="silver",
                        if_exists="replace", index=False)
        print(f"  Saved → postgresql://silver.{table}")

    engine.dispose()

if __name__ == "__main__":
    run_silver_transform()