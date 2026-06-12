import sys
sys.path.insert(0, "/workspaces/multi-agent-rag")

import os
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from agents.state import State
from agents.router import create_primary_assistant
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

# Tất cả tools
ALL_TOOLS = [
    search_flights,
    lookup_passenger,
    search_hotels,
    search_car_rentals,
    search_trip_recommendations,
    book_hotel,
    book_car_rental,
    update_flight,
]

SENSITIVE_TOOL_NAMES = {
    "book_hotel",
    "book_car_rental",
    "update_flight",
}

def route_tools(state: State):
    """Router: nếu tool là sensitive → human_review, ngược lại → tools."""
    next_node = tools_condition(state)
    if next_node == END:
        return END
    # Kiểm tra tool nào được gọi
    ai_message = state["messages"][-1]
    first_tool = ai_message.tool_calls[0]["name"]
    if first_tool in SENSITIVE_TOOL_NAMES:
        return "human_review"
    return "tools"

def human_review_node(state: State):
    """Interrupt để user confirm trước khi chạy sensitive tool."""
    pass  # LangGraph interrupt tự handle khi có MemorySaver

def fetch_user_info(state: State):
    """Load passenger info vào state khi bắt đầu conversation."""
    return {"user_info": "Passenger: John Doe | Ticket: 0005432000987"}

def assistant_node(state: State):
    assistant = create_primary_assistant(ALL_TOOLS)
    result = assistant.invoke({
        "messages": state["messages"],
        "user_info": state.get("user_info", "Unknown passenger"),
        "time": datetime.now().isoformat(),
    })
    return {"messages": result}

def build_graph():
    builder = StateGraph(State)

    # Nodes
    builder.add_node("fetch_user_info", fetch_user_info)
    builder.add_node("assistant",       assistant_node)
    builder.add_node("tools",           ToolNode(ALL_TOOLS))
    builder.add_node("human_review",    human_review_node)

    # Edges
    builder.add_edge(START,            "fetch_user_info")
    builder.add_edge("fetch_user_info", "assistant")
    builder.add_conditional_edges("assistant", route_tools)
    builder.add_edge("tools",           "assistant")
    builder.add_edge("human_review",    "assistant")

    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["human_review"],
    )

graph = build_graph()

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-1"}}

    print("=" * 50)
    print("Travel RAG Assistant")
    print("=" * 50)

    def get_response_text(result):
        """Extract text từ response dù là string hay list of dict."""
        last = result["messages"][-1]
        if hasattr(last, "content"):
            content = last.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    block.get("text", "") 
                    for block in content 
                    if isinstance(block, dict)
                )
        return str(last)

    print("\nQuery 1: Trip recommendations")
    result = graph.invoke(
        {"messages": [HumanMessage(content="What historic landmarks can I visit in Basel?")]},
        config=config,
    )
    print(get_response_text(result))

    print("\nQuery 2: Flight search")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Find flights from ZRH to MUC")]},
        config=config,
    )
    print(get_response_text(result))

    print("\nQuery 3: Hotel search")
    result = graph.invoke(
        {"messages": [HumanMessage(content="Find me hotels in Basel")]},
        config=config,
    )
    print(get_response_text(result))