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

WORKFLOW — FOLLOW EXACTLY:
1. Call search_flights ONCE to find available flights.
2. After receiving results, present them clearly to the user.
3. If user wants to update a flight, call update_flight (requires confirmation).
4. After presenting results OR completing action, call leave_skill.

RULES:
- Call search_flights only ONCE. Never call it twice.
- Always present results as text before calling leave_skill.
- ONLY use data returned by tools. Never invent flight info.

Current time: {time}
Passenger: {user_info}""",
    ),
    ("placeholder", "{messages}"),
])

CAR_RENTAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized car rental assistant.

WORKFLOW — FOLLOW EXACTLY:
1. Call search_car_rentals ONCE to find available cars.
2. After receiving results, present them clearly to the user.
3. If user wants to book, call book_car_rental (requires confirmation).
4. After presenting results OR completing booking, call leave_skill.

RULES:
- Call search_car_rentals only ONCE. Never call it twice.
- Always present results as text before calling leave_skill.
- ONLY use data returned by tools. Never invent car rental info.

Current time: {time}
Passenger: {user_info}""",
    ),
    ("placeholder", "{messages}"),
])

EXCURSION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a specialized excursion and trip recommendation assistant.

WORKFLOW — FOLLOW EXACTLY:
1. Call search_trip_recommendations ONCE with a relevant query.
2. After receiving results, present them clearly to the user.
3. After presenting results, call leave_skill.

RULES:
- Call search_trip_recommendations only ONCE.
- Always present results as text before calling leave_skill.
- ONLY recommend places returned by the tool. Never invent destinations.

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

WORKFLOW — FOLLOW EXACTLY:
1. Call search_hotels ONCE to find available hotels.
2. After receiving search results, present them to the user in a clear format.
3. Ask if they want to book any hotel.
4. If user confirms booking, call book_hotel (requires confirmation).
5. After presenting results OR completing booking, call leave_skill.

RULES:
- Call search_hotels only ONCE. Never call it twice.
- Always present the results as text before calling leave_skill.
- ONLY use data returned by tools. Never invent hotel info.
- Format response clearly with hotel names, price tiers, and dates.

Current time: {time}
Passenger: {user_info}""",
    ),
    ("placeholder", "{messages}"),
])

def create_hotel_assistant(tools: list):
    return HOTEL_PROMPT | get_llm().bind_tools(tools + [leave_skill])

# ── Car Rental Assistant ─────────────────────────────────────

# CAR_RENTAL_PROMPT = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """You are a specialized car rental assistant.
# You help users find and book car rentals.

# RULES:
# 1. Only handle car rental requests.
# 2. Use search_car_rentals to find available cars.
# 3. Use book_car_rental to reserve a car (requires user confirmation).
# 4. When done, call leave_skill to return to the primary assistant.
# 5. ONLY use data returned by tools. Never invent car rental info.

# Current time: {time}
# Passenger: {user_info}""",
#     ),
#     ("placeholder", "{messages}"),
# ])

def create_car_rental_assistant(tools: list):
    return CAR_RENTAL_PROMPT | get_llm().bind_tools(tools + [leave_skill])

# ── Excursion Assistant ──────────────────────────────────────

# EXCURSION_PROMPT = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """You are a specialized excursion and trip recommendation assistant.
# You help users discover activities, landmarks, and attractions.

# RULES:
# 1. Only handle excursion and activity requests.
# 2. ALWAYS call search_trip_recommendations before answering.
# 3. ONLY recommend places returned by the tool. Never invent destinations.
# 4. When done, call leave_skill to return to the primary assistant.

# Current time: {time}
# Passenger: {user_info}""",
#     ),
#     ("placeholder", "{messages}"),
# ])

def create_excursion_assistant(tools: list):
    return EXCURSION_PROMPT | get_llm().bind_tools(tools + [leave_skill])