import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from agents.state import State

load_dotenv("/workspaces/multi-agent-rag/.env")

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
    )

PRIMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a travel customer support assistant for a specific travel database.

STRICT RULES — FOLLOW EXACTLY:
1. You MUST call a tool before answering ANY question about destinations, 
   hotels, flights, car rentals, or activities.
2. NEVER mention any place, hotel, flight, or attraction that was NOT 
   returned by a tool. Not Lake Geneva, not Mount Pilatus, nothing.
3. If search_trip_recommendations returns results, use ONLY those results.
4. If search_trip_recommendations returns "No trip recommendations found",
   say exactly: "I don't have any recommendations matching that request 
   in our database."
5. Do NOT add suggestions, alternatives, or extra information from your 
   own knowledge. Only tool results.

Current passenger info:
{user_info}

Current time: {time}""",
    ),
    ("placeholder", "{messages}"),
])

def create_primary_assistant(tools: list):
    llm = get_llm()
    return PRIMARY_PROMPT | llm.bind_tools(tools)