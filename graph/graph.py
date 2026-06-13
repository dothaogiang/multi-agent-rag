import sys
sys.path.insert(0, "/workspaces/multi-agent-rag")

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, ToolMessage

from agents.state import State
from agents.router import (
    create_primary_assistant,
    to_flight_assistant,
    to_hotel_assistant,
    to_car_rental_assistant,
    to_excursion_assistant,
    HANDOFF_TOOLS,
)
from agents.specialized_agents import (
    create_flight_assistant,
    create_hotel_assistant,
    create_car_rental_assistant,
    create_excursion_assistant,
    leave_skill,
)
from tools.safe_tools import (
    search_flights,
    lookup_passenger,
    search_hotels,
    search_car_rentals,
    search_trip_recommendations,
)
from tools.sensitive_tools import (
    book_hotel,
    book_car_rental,
    update_flight,
)

load_dotenv("/workspaces/multi-agent-rag/.env")

os.environ["LANGSMITH_TRACING"]   = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_ENDPOINT"]  = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_API_KEY"]   = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"]   = os.getenv("LANGSMITH_PROJECT", "/multi-agent-rag")

# ── Tool sets per agent ───────────────────────────────────────

FLIGHT_TOOLS      = [search_flights, lookup_passenger, update_flight]
HOTEL_TOOLS       = [search_hotels, book_hotel]
CAR_RENTAL_TOOLS  = [search_car_rentals, book_car_rental]
EXCURSION_TOOLS   = [search_trip_recommendations]

SENSITIVE_TOOL_NAMES = {"book_hotel", "book_car_rental", "update_flight"}

# ── Helper: get common invoke args ───────────────────────────

def get_invoke_args(state: State) -> dict:
    return {
        "messages":   state["messages"],
        "user_info":  state.get("user_info", "Unknown passenger"),
        "time":       datetime.now().isoformat(),
    }

# ── Node functions ────────────────────────────────────────────

def fetch_user_info(state: State):
    return {"user_info": "Passenger: John Doe | Ticket: 0005432000987"}

def primary_assistant_node(state: State):
    assistant = create_primary_assistant()
    result    = assistant.invoke(get_invoke_args(state))
    return {"messages": result, "active_agent": "primary"}

def flight_assistant_node(state: State):
    assistant = create_flight_assistant(FLIGHT_TOOLS)
    result    = assistant.invoke(get_invoke_args(state))
    return {"messages": result, "active_agent": "flight"}

def hotel_assistant_node(state: State):
    assistant = create_hotel_assistant(HOTEL_TOOLS)
    result    = assistant.invoke(get_invoke_args(state))
    return {"messages": result, "active_agent": "hotel"}

def car_rental_assistant_node(state: State):
    assistant = create_car_rental_assistant(CAR_RENTAL_TOOLS)
    result    = assistant.invoke(get_invoke_args(state))
    return {"messages": result, "active_agent": "car_rental"}

def excursion_assistant_node(state: State):
    assistant = create_excursion_assistant(EXCURSION_TOOLS)
    result    = assistant.invoke(get_invoke_args(state))
    return {"messages": result, "active_agent": "excursion"}

def human_review_node(state: State):
    """Interrupt point — LangGraph tự handle với interrupt_before."""
    pass

# ── Routing functions ─────────────────────────────────────────

def route_primary(state: State):
    """Router: primary assistant → specialized agent."""
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END
    tool_name = last.tool_calls[0]["name"]
    routes = {
        "to_flight_assistant":     "flight_assistant",
        "to_hotel_assistant":      "hotel_assistant",
        "to_car_rental_assistant": "car_rental_assistant",
        "to_excursion_assistant":  "excursion_assistant",
    }
    return routes.get(tool_name, END)

def create_entry_node(agent_name: str, tool_name: str):
    """Tạo node inject ToolMessage để specialized agent biết context."""
    def entry_node(state: State):
        # Lấy tool_call_id từ message trước
        last = state["messages"][-1]
        tool_call_id = last.tool_calls[0]["id"] if hasattr(last, "tool_calls") and last.tool_calls else str(uuid.uuid4())
        
        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {agent_name}. "
                            f"Reflect on the above conversation and assist "
                            f"the user with their {agent_name.lower()} needs.",
                    tool_call_id=tool_call_id,
                )
            ],
            "active_agent": agent_name.lower().replace(" ", "_"),
        }
    return entry_node

def route_specialized(state: State):
    """Router: specialized agent → tools / human_review / primary."""
    last = state["messages"][-1]
    if not hasattr(last, "tool_calls") or not last.tool_calls:
        return END
    tool_name = last.tool_calls[0]["name"]
    if tool_name == "leave_skill":
        return "primary_assistant"
    if tool_name in SENSITIVE_TOOL_NAMES:
        return "human_review"
    return "tools"

def create_tool_node_with_fallback(tools: list) -> ToolNode:
    return ToolNode(tools).with_fallbacks(
        [ToolNode(tools)],
        exception_key="error",
    )

# ── Build graph ───────────────────────────────────────────────

def build_graph():
    builder = StateGraph(State)

    ALL_TOOLS = (
        FLIGHT_TOOLS + HOTEL_TOOLS + CAR_RENTAL_TOOLS +
        EXCURSION_TOOLS + HANDOFF_TOOLS + [leave_skill]
    )

    # Entry nodes cho mỗi specialized agent
    builder.add_node("enter_flight",     create_entry_node("Flight Assistant",     "to_flight_assistant"))
    builder.add_node("enter_hotel",      create_entry_node("Hotel Assistant",      "to_hotel_assistant"))
    builder.add_node("enter_car_rental", create_entry_node("Car Rental Assistant", "to_car_rental_assistant"))
    builder.add_node("enter_excursion",  create_entry_node("Excursion Assistant",  "to_excursion_assistant"))

    # Specialized agent nodes
    builder.add_node("fetch_user_info",       fetch_user_info)
    builder.add_node("primary_assistant",     primary_assistant_node)
    builder.add_node("flight_assistant",      flight_assistant_node)
    builder.add_node("hotel_assistant",       hotel_assistant_node)
    builder.add_node("car_rental_assistant",  car_rental_assistant_node)
    builder.add_node("excursion_assistant",   excursion_assistant_node)
    builder.add_node("tools",                 ToolNode(ALL_TOOLS))
    builder.add_node("human_review",          human_review_node)

    # Entry
    builder.add_edge(START, "fetch_user_info")
    builder.add_edge("fetch_user_info", "primary_assistant")

    # Primary → entry nodes (không direct sang agent)
    builder.add_conditional_edges(
        "primary_assistant",
        route_primary,
        {
            "flight_assistant":     "enter_flight",
            "hotel_assistant":      "enter_hotel",
            "car_rental_assistant": "enter_car_rental",
            "excursion_assistant":  "enter_excursion",
            END:                    END,
        },
    )

    # Entry nodes → specialized agents
    builder.add_edge("enter_flight",     "flight_assistant")
    builder.add_edge("enter_hotel",      "hotel_assistant")
    builder.add_edge("enter_car_rental", "car_rental_assistant")
    builder.add_edge("enter_excursion",  "excursion_assistant")

    # Specialized agents → tools / human_review / back to primary
    for agent in ["flight_assistant", "hotel_assistant",
                  "car_rental_assistant", "excursion_assistant"]:
        builder.add_conditional_edges(
            agent,
            route_specialized,
            {
                "tools":             "tools",
                "human_review":      "human_review",
                "primary_assistant": "primary_assistant",
                END:                 END,
            },
        )

    # Tools → back to active specialized agent
    builder.add_conditional_edges(
        "tools",
        lambda state: state.get("active_agent", "primary") + "_assistant"
        if state.get("active_agent") not in ("primary", None)
        else "primary_assistant",
        {
            "flight_assistant":     "flight_assistant",
            "hotel_assistant":      "hotel_assistant",
            "car_rental_assistant": "car_rental_assistant",
            "excursion_assistant":  "excursion_assistant",
            "primary_assistant":    "primary_assistant",
        },
    )

    # human_review → back to active agent
    builder.add_conditional_edges(
        "human_review",
        lambda state: state.get("active_agent", "primary") + "_assistant"
        if state.get("active_agent") not in ("primary", None)
        else "primary_assistant",
        {
            "flight_assistant":     "flight_assistant",
            "hotel_assistant":      "hotel_assistant",
            "car_rental_assistant": "car_rental_assistant",
            "excursion_assistant":  "excursion_assistant",
            "primary_assistant":    "primary_assistant",
        },
    )

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],
    )

graph = build_graph()


def get_active_agent(result):
    """Lấy agent cuối cùng thật sự xử lý query — bỏ qua primary."""
    agent_map = {
        "to_flight_assistant":     "flight",
        "to_hotel_assistant":      "hotel",
        "to_car_rental_assistant": "car_rental",
        "to_excursion_assistant":  "excursion",
    }
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] in agent_map:
                    return agent_map[tc["name"]]
    return "primary"

# ── Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    def get_response(result):
        """Lấy AIMessage có content thật — kể cả khi message có tool_calls như leave_skill."""
        for msg in reversed(result["messages"]):
            if msg.__class__.__name__ != "AIMessage":
                continue
            content = msg.content
            # Xử lý list content
            if isinstance(content, list):
                text = " ".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            elif isinstance(content, str):
                text = content
            else:
                continue
            if text.strip():
                return text
        return "(no response)"


    print("=" * 60)
    print("Multi-Agent Travel Assistant")
    print("=" * 60)

    config = {"configurable": {"thread_id": "specialized-test-1"}}

    queries = [
        "Find hotels in Basel",
        "What can I visit in Zurich?",
        "Search flights from ZRH to FRA",
        "I need a car rental in Lucerne",
    ]

    for q in queries:
        print(f"\n👤 {q}")
        result = graph.invoke(
            {"messages": [HumanMessage(content=q)]},
            config={"configurable": {"thread_id": f"test-{q[:10]}"}},
        )
        # Show which agent handled it
        active = get_active_agent(result)
        print(f"[Agent: {active}]")
        print(f"🤖 {get_response(result)[:150]}")