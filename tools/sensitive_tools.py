from sqlalchemy import text
from langchain_core.tools import tool
from tools.safe_tools import get_engine
from tools.base import handle_tool_error
from tools.safe_tools import (
    search_flights as _safe_search_flights,
    search_hotels as _safe_search_hotels,
)

@tool
def book_hotel(hotel_id: int) -> str:
    """Book a hotel by id. Requires user confirmation."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT name, location, booked FROM silver.hotels WHERE id = :id"),
            {"id": hotel_id}
        ).fetchone()
        if not result:
            return f"Hotel {hotel_id} not found."
        if result.booked:
            return f"Hotel '{result.name}' is already booked."
        conn.execute(
            text("UPDATE silver.hotels SET booked = true WHERE id = :id"),
            {"id": hotel_id}
        )
    engine.dispose()
    return f"Successfully booked hotel '{result.name}' in {result.location}."


@tool
def book_car_rental(rental_id: int) -> str:
    """Book a car rental by id. Requires user confirmation."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT name, location, booked FROM silver.car_rentals WHERE id = :id"),
            {"id": rental_id}
        ).fetchone()
        if not result:
            return f"Car rental {rental_id} not found."
        if result.booked:
            return f"Car rental '{result.name}' is already booked."
        conn.execute(
            text("UPDATE silver.car_rentals SET booked = true WHERE id = :id"),
            {"id": rental_id}
        )
    engine.dispose()
    return f"Successfully booked car rental '{result.name}' in {result.location}."


@tool
def update_flight(ticket_no: str, new_flight_id: int) -> str:
    """Update passenger flight booking. Requires user confirmation."""
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT flight_no FROM silver.flights WHERE flight_id = :id"),
            {"id": new_flight_id}
        ).fetchone()
        if not result:
            return f"Flight {new_flight_id} not found."
        conn.execute(
            text("UPDATE silver.flights SET ticket_no = :ticket WHERE flight_id = :id"),
            {"ticket": ticket_no, "id": new_flight_id}
        )
    engine.dispose()
    return f"Flight updated to {result.flight_no} for ticket {ticket_no}."

@tool
@handle_tool_error
def search_flights(departure_airport: str, arrival_airport: str, limit: int = 5) -> str:
    """Search available flights between two airports (proxy to safe_tools)."""
    # Proxy to the implementation in safe_tools
    return _safe_search_flights(departure_airport, arrival_airport, limit)

@tool
@handle_tool_error
def search_hotels(location: str, limit: int = 5) -> str:
    """Search available hotels in a given location (proxy to safe_tools)."""
    # Proxy to the implementation in safe_tools
    return _safe_search_hotels(location, limit)