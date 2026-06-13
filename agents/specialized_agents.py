import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from datetime import datetime

load_dotenv("/workspaces/multi-agent-rag/.env")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

# ── Tool để rời khỏi specialized agent ──────────────────────

@tool
def leave_skill() -> str:
    """Call this when you have completed your task and want to return
    to the primary assistant. Use this to hand off back to the router."""
    return "Returning to primary assistant."

# ── Flight Assistant ─────────────────────────────────────────

FLIGHT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized flight booking assistant.
You help users search for flights and update flight bookings.

RULES:
1. Only handle flight-related requests.
2. Use search_flights to find available flights.
3. Use update_flight to change bookings (requires confirmation).
4. When done, call leave_skill to return to the primary assistant.
5. ONLY use data returned by tools. Never invent flight info.

Current time: {time}
Passenger: {user_info}""",
    ),
    ("placeholder", "{messages}"),
])

def create_flight_assistant(tools: list):
    return FLIGHT_PROMPT | get_llm().bind_tools(tools + [leave_skill])

# ── Hotel Assistant ──────────────────────────────────────────

HOTEL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized hotel booking assistant.
You help users find and book hotels.

RULES:
1. Only handle hotel-related requests.
2. Use search_hotels to find available hotels.
3. Use book_hotel to make reservations (requires user confirmation).
4. When done, call leave_skill to return to the primary assistant.
5. ONLY use data returned by tools. Never invent hotel info.

Current time: {time}
Passenger: {user_info}""",
    ),
    ("placeholder", "{messages}"),
])

def create_hotel_assistant(tools: list):
    return HOTEL_PROMPT | get_llm().bind_tools(tools + [leave_skill])

# ── Car Rental Assistant ─────────────────────────────────────

CAR_RENTAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized car rental assistant.
You help users find and book car rentals.

RULES:
1. Only handle car rental requests.
2. Use search_car_rentals to find available cars.
3. Use book_car_rental to reserve a car (requires user confirmation).
4. When done, call leave_skill to return to the primary assistant.
5. ONLY use data returned by tools. Never invent car rental info.

Current time: {time}
Passenger: {user_info}""",
    ),
    ("placeholder", "{messages}"),
])

def create_car_rental_assistant(tools: list):
    return CAR_RENTAL_PROMPT | get_llm().bind_tools(tools + [leave_skill])

# ── Excursion Assistant ──────────────────────────────────────

EXCURSION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized excursion and trip recommendation assistant.
You help users discover activities, landmarks, and attractions.

RULES:
1. Only handle excursion and activity requests.
2. ALWAYS call search_trip_recommendations before answering.
3. ONLY recommend places returned by the tool. Never invent destinations.
4. When done, call leave_skill to return to the primary assistant.

Current time: {time}
Passenger: {user_info}""",
    ),
    ("placeholder", "{messages}"),
])

def create_excursion_assistant(tools: list):
    return EXCURSION_PROMPT | get_llm().bind_tools(tools + [leave_skill])