import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

load_dotenv()

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://travel_user:travel_pass@localhost:5432/travel_db"
)

def get_connection():
    return psycopg2.connect(POSTGRES_URL)

def create_schemas():
    conn = get_connection()
    cur  = conn.cursor()
    for schema in ["bronze", "silver", "gold"]:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                sql.Identifier(schema)
            )
        )
        print(f"  Schema '{schema}' ready.")
    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    create_schemas()