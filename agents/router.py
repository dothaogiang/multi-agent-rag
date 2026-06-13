import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv("/workspaces/multi-agent-rag/.env")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

# ── Handoff tools — để router chuyển sang specialized agent ──

@tool
def to_flight_assistant() -> str:
    """Transfer to the flight booking assistant for flight searches
    and flight booking updates."""
    return "Transferring to flight assistant."

@tool
def to_hotel_assistant() -> str:
    """Transfer to the hotel booking assistant for hotel searches
    and hotel reservations."""
    return "Transferring to hotel assistant."

@tool
def to_car_rental_assistant() -> str:
    """Transfer to the car rental assistant for car rental searches
    and car reservations."""
    return "Transferring to car rental assistant."

@tool
def to_excursion_assistant() -> str:
    """Transfer to the excursion assistant for trip recommendations,
    landmarks, activities, and things to do."""
    return "Transferring to excursion assistant."

# ── Primary assistant prompt ──────────────────────────────────

PRIMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a travel customer support router.
Your job is to understand the user's intent and delegate to the 
right specialized assistant.

ROUTING RULES — STRICT:
- Flights (search, update)     → to_flight_assistant
- Hotels (search, book)        → to_hotel_assistant
- Car rentals (search, book)   → to_car_rental_assistant
- Activities, landmarks        → to_excursion_assistant

MULTI-REQUEST HANDLING:
- If user asks for BOTH hotel AND car rental → call to_hotel_assistant first
- After hotel_assistant returns, if car rental was also requested → call to_car_rental_assistant
- Handle one agent at a time, sequentially

IMPORTANT:
- ALWAYS delegate. Never answer travel questions yourself.
- For non-travel questions, respond politely and redirect.

Current passenger: {user_info}
Current time: {time}""",
    ),
    ("placeholder", "{messages}"),
])

HANDOFF_TOOLS = [
    to_flight_assistant,
    to_hotel_assistant,
    to_car_rental_assistant,
    to_excursion_assistant,
]

def create_primary_assistant():
    return PRIMARY_PROMPT | get_llm().bind_tools(HANDOFF_TOOLS)