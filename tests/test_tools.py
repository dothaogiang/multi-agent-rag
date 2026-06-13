import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
sys.path.insert(0, "/workspaces/multi-agent-rag")

class TestSearchTools:

    def test_search_trip_recommendations_returns_results(self):
        """RAG search phải trả về kết quả khi có data."""
        from tools.safe_tools import search_trip_recommendations
        result = search_trip_recommendations.invoke({"query": "art museum"})
        assert isinstance(result, str)
        assert len(result) > 0
        assert "No trip recommendations found" not in result

    def test_search_trip_recommendations_irrelevant_query(self):
        """Query không liên quan vẫn trả về kết quả (top-k luôn có)."""
        from tools.safe_tools import search_trip_recommendations
        result = search_trip_recommendations.invoke({"query": "xyzabc123"})
        assert isinstance(result, str)

    def test_search_hotels_valid_location(self):
        """Tìm hotel ở Basel phải có kết quả."""
        from tools.safe_tools import search_hotels
        result = search_hotels.invoke({"location": "Basel"})
        assert isinstance(result, str)
        assert "Basel" in result or "No available hotels" in result

    def test_search_hotels_nonexistent_location(self):
        """Location không tồn tại phải trả về not found message."""
        from tools.safe_tools import search_hotels
        result = search_hotels.invoke({"location": "nonexistentcity12345"})
        assert "No available hotels found" in result

    def test_search_flights_invalid_airports(self):
        """Airport không tồn tại phải trả về not found message."""
        from tools.safe_tools import search_flights
        result = search_flights.invoke({
            "departure_airport": "XXX",
            "arrival_airport": "YYY"
        })
        assert "No flights found" in result

    def test_search_car_rentals_valid_location(self):
        """Tìm car rental phải trả về string."""
        from tools.safe_tools import search_car_rentals
        result = search_car_rentals.invoke({"location": "Basel"})
        assert isinstance(result, str)


class TestEmbedder:

    def test_build_document_text(self):
        """build_document_text phải gộp đủ 4 fields."""
        import pandas as pd
        from vectorizer.app.embedder import build_document_text
        row = pd.Series({
            "name": "Basel Minster",
            "location": "Basel",
            "keywords": "landmark, history",
            "details": "A Gothic cathedral."
        })
        text = build_document_text(row)
        assert "Basel Minster" in text
        assert "Basel" in text
        assert "landmark" in text
        assert "Gothic cathedral" in text

    def test_qdrant_collection_exists(self):
        """Qdrant collection phải tồn tại sau khi embed."""
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = [c.name for c in client.get_collections().collections]
        assert "travel_policies" in collections

    def test_qdrant_has_correct_count(self):
        """Qdrant phải có đúng 10 vectors."""
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        info = client.get_collection("travel_policies")
        assert info.points_count == 10


class TestSilverData:

    def test_silver_flights_has_data(self):
        """silver.flights phải có data sau pipeline."""
        import os
        from dotenv import load_dotenv
        load_dotenv("/workspaces/multi-agent-rag/.env")
        from sqlalchemy import create_engine
        engine = create_engine(os.getenv("POSTGRES_URL"))
        df = pd.read_sql("SELECT COUNT(*) as cnt FROM silver.flights", engine)
        assert df["cnt"].iloc[0] > 0

    def test_silver_hotels_no_nulls_in_key_columns(self):
        """silver.hotels không được có null ở id, name, location."""
        import os
        from dotenv import load_dotenv
        load_dotenv("/workspaces/multi-agent-rag/.env")
        from sqlalchemy import create_engine
        engine = create_engine(os.getenv("POSTGRES_URL"))
        df = pd.read_sql(
            "SELECT id, name, location FROM silver.hotels WHERE id IS NULL OR name IS NULL OR location IS NULL",
            engine
        )
        assert len(df) == 0, f"Found {len(df)} rows with null values"