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
        """You are a helpful travel customer support assistant.
Your job is to help passengers with:
- Flight information and updates
- Hotel bookings
- Car rentals
- Trip recommendations and excursions

Current passenger info:
{user_info}

Use the available tools to look up information and assist the passenger.
For bookings that modify data (book hotel, book car, update flight),
always confirm with the passenger before proceeding.

Current time: {time}""",
    ),
    ("placeholder", "{messages}"),
])

def create_primary_assistant(tools: list):
    llm = get_llm()
    return PRIMARY_PROMPT | llm.bind_tools(tools)