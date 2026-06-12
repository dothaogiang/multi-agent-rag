import os
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv("/workspaces/multi-agent-rag/.env")

QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "travel_policies")
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
POSTGRES_URL    = os.getenv(
    "POSTGRES_URL",
    "postgresql://travel_user:travel_pass@localhost:5432/travel_db"
)
EMBED_MODEL = "gemini-embedding-001"

def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

def embed_texts(client, texts: list[str]) -> list[list[float]]:
    """Embed danh sách texts dùng google.genai SDK mới."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in response.embeddings]

def embed_query(client, query: str) -> list[float]:
    """Embed 1 query để search."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values

def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        print(f"  Created collection: {collection_name}")
    else:
        print(f"  Collection already exists: {collection_name}")

def load_trip_recommendations() -> pd.DataFrame:
    from sqlalchemy import create_engine
    engine = create_engine(POSTGRES_URL)
    df = pd.read_sql("SELECT * FROM silver.trip_recommendations", engine)
    engine.dispose()
    return df

def build_document_text(row: pd.Series) -> str:
    return (
        f"Destination: {row['name']}\n"
        f"Location: {row['location']}\n"
        f"Keywords: {row['keywords']}\n"
        f"Details: {row['details']}"
    )

def run_embedding():
    print("Loading trip recommendations from PostgreSQL silver...")
    df = load_trip_recommendations()
    print(f"  Loaded {len(df)} records")

    texts = [build_document_text(row) for _, row in df.iterrows()]
    print(f"  Built {len(texts)} documents to embed")

    print(f"Generating embeddings via {EMBED_MODEL}...")
    genai_client = get_genai_client()
    vectors = embed_texts(genai_client, texts)
    vector_size = len(vectors[0])
    print(f"  Vector size: {vector_size}")

    qdrant = get_qdrant_client()
    create_collection(qdrant, COLLECTION_NAME, vector_size)

    points = [
        PointStruct(
            id=int(df.iloc[i]["id"]),
            vector=vectors[i],
            payload={
                "name":     df.iloc[i]["name"],
                "location": df.iloc[i]["location"],
                "keywords": df.iloc[i]["keywords"],
                "details":  df.iloc[i]["details"],
                "booked":   bool(df.iloc[i]["booked"]),
                "text":     texts[i],
            },
        )
        for i in range(len(df))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Upserted {len(points)} vectors → Qdrant '{COLLECTION_NAME}'")

if __name__ == "__main__":
    run_embedding()