import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from langchain_core.tools import tool
from tools.base import handle_tool_error
from sqlalchemy import create_engine, event
from sqlalchemy.pool import QueuePool

load_dotenv("/workspaces/multi-agent-rag/.env")

POSTGRES_URL    = os.getenv("POSTGRES_URL",
    "postgresql://travel_user:travel_pass@localhost:5432/travel_db")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "travel_policies")

def get_qdrant():
    return QdrantClient(host="localhost", port=6333)

def get_genai():
    return genai.Client(api_key=GOOGLE_API_KEY)

# ── Flight tools ──────────────────────────────────────────────

@tool
@handle_tool_error
def search_flights(
    departure_airport: str,
    arrival_airport: str,
    limit: int = 5,
) -> str:
    """Search available flights between two airports."""
    engine = get_engine()
    query = text("""
        SELECT DISTINCT
            flight_no,
            departure_airport,
            arrival_airport,
            scheduled_departure,
            scheduled_arrival,
            status,
            aircraft_code
        FROM silver.flights
        WHERE departure_airport = :dep
          AND arrival_airport   = :arr
          AND status != 'Cancelled'
        ORDER BY scheduled_departure
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "dep": departure_airport.upper(),
            "arr": arrival_airport.upper(),
            "limit": limit,
        })
    engine.dispose()
    if df.empty:
        return f"No flights found from {departure_airport} to {arrival_airport}."
    return df.to_string(index=False)

@tool
@handle_tool_error
def lookup_passenger(passenger_id: str) -> str:
    """Look up flight info by ticket number or flight number."""
    engine = get_engine()
    query = text("""
        SELECT
            flight_no,
            departure_airport,
            arrival_airport,
            scheduled_departure,
            scheduled_arrival,
            status,
            aircraft_code
        FROM silver.flights
        WHERE flight_no = :flight_no
        LIMIT 5
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"flight_no": passenger_id.upper()})
    engine.dispose()
    if df.empty:
        return f"No flight found for: {passenger_id}"
    return df.to_string(index=False)

@tool
@handle_tool_error
def search_hotels(location: str, limit: int = 5) -> str:
    """Search available hotels in a given location."""
    engine = get_engine()
    query = text("""
        SELECT id, name, location, price_tier,
               checkin_date, checkout_date
        FROM silver.hotels
        WHERE LOWER(location) LIKE :loc
          AND booked = false
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "loc": f"%{location.lower()}%",
            "limit": limit,
        })
    engine.dispose()
    if df.empty:
        return f"No available hotels found in {location}."
    return df.to_string(index=False)


@tool
@handle_tool_error
def search_car_rentals(location: str, limit: int = 5) -> str:
    """Search available car rentals in a given location."""
    engine = get_engine()
    query = text("""
        SELECT id, name, location, price_tier,
               start_date, end_date
        FROM silver.car_rentals
        WHERE LOWER(location) LIKE :loc
          AND booked = false
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "loc": f"%{location.lower()}%",
            "limit": limit,
        })
    engine.dispose()
    if df.empty:
        return f"No car rentals available in {location}."
    return df.to_string(index=False)


@tool
@handle_tool_error
def search_trip_recommendations(query: str, limit: int = 3) -> str:
    """Search trip recommendations and excursions using semantic search."""
    genai_client = get_genai()
    response = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    vector = response.embeddings[0].values

    qdrant = get_qdrant()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
    ).points

    if not results:
        return "No trip recommendations found."

    output = []
    for r in results:
        output.append(
            f"- {r.payload['name']} ({r.payload['location']})\n"
            f"  Keywords: {r.payload['keywords']}\n"
            f"  {r.payload['details']}"
        )
    return "\n\n".join(output)


def get_engine():
    """Engine với connection pool và retry."""
    return create_engine(
        POSTGRES_URL,
        poolclass=QueuePool,
        pool_size=5,
        pool_timeout=10,
        pool_pre_ping=True,   # ← test connection trước khi dùng
        connect_args={"connect_timeout": 5},
    )