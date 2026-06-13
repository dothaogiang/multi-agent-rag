import sys
sys.path.insert(0, "/workspaces/multi-agent-rag")

from langchain_core.messages import HumanMessage
from graph.graph import graph, get_response, get_active_agent

def run(query: str, thread_id: str) -> tuple[str, str]:
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    return get_active_agent(result), get_response(result)

def resume(query: str, thread_id: str) -> tuple[str, str]:
    """Tiếp tục graph sau khi bị interrupt."""
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    return get_active_agent(result), get_response(result)

print("=" * 60)
print("DEMO 1: Agent Routing — đúng agent nhận đúng query")
print("=" * 60)

tests = [
    ("Find hotels in Basel",             "d1-hotel"),
    ("What can I visit in Zurich?",      "d1-excursion"),
    ("Search flights from BSL to MCO",   "d1-flight"),
    ("I need a car rental in Lucerne",   "d1-car"),
]
for query, tid in tests:
    agent, response = run(query, tid)
    print(f"\n👤 {query}")
    print(f"   [→ {agent}_assistant]")
    print(f"   🤖 {response[:120]}")

print()
print("=" * 60)
print("DEMO 2: Multi-turn Memory — nhớ context qua nhiều turns")
print("=" * 60)

# Turn 1
agent, resp = run("Find hotels in Basel", "d2-memory")
print(f"\n👤 Turn 1: Find hotels in Basel")
print(f"   [→ {agent}_assistant]")
print(f"   🤖 {resp[:120]}")

# Turn 2 — nhớ đang nói về Basel hotels
agent, resp = run("Which one is cheapest?", "d2-memory")
print(f"\n👤 Turn 2: Which one is cheapest?")
print(f"   [→ {agent}_assistant]")
print(f"   🤖 {resp[:120]}")

# Turn 3 — chuyển sang excursion
agent, resp = run("What can I do near there?", "d2-memory")
print(f"\n👤 Turn 3: What can I do near there?")
print(f"   [→ {agent}_assistant]")
print(f"   🤖 {resp[:120]}")

print()
print("=" * 60)
print("DEMO 3: Human-in-the-loop — dừng lại trước khi book")
print("=" * 60)

from graph.graph import graph as g

config = {"configurable": {"thread_id": "d3-booking"}}

# Turn 1: tìm hotel
r1 = g.invoke(
    {"messages": [HumanMessage(content="Find hotels in Basel")]},
    config=config,
)
print(f"\n👤 Find hotels in Basel")
print(f"   🤖 {get_response(r1)[:120]}")

# Turn 2: yêu cầu book → graph bị interrupt
r2 = g.invoke(
    {"messages": [HumanMessage(content="Book the Hilton Basel")]},
    config=config,
)
state = g.get_state(config)
print(f"\n👤 Book the Hilton Basel")
print(f"   [Graph state: next={state.next}]")

if state.next == ("human_review",):
    # Tìm pending tool call
    for msg in state.values["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "book_hotel":
                    print(f"   ⚠️  INTERRUPTED — waiting for confirmation")
                    print(f"   Tool: book_hotel | Args: {tc['args']}")

    # User confirm
    confirm = input("\n👤 Confirm booking? (yes/no): ").strip().lower()
    r3 = g.invoke(
        {"messages": [HumanMessage(content=confirm)]},
        config=config,
    )
    print(f"   🤖 {get_response(r3)[:150]}")

print()
print("=" * 60)
print("DEMO 4: Parallel tools — gọi nhiều tools cùng lúc")
print("=" * 60)

r4_config = {"configurable": {"thread_id": "d4-parallel"}}
r4 = g.invoke(
    {"messages": [HumanMessage(
        content="Plan my Basel trip: find a hotel and car rental"
    )]},
    config=r4_config,
)

tools_called = []
for msg in r4["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            if tc["name"] not in ("to_hotel_assistant", "to_car_rental_assistant",
                                   "leave_skill", "to_flight_assistant",
                                   "to_excursion_assistant"):
                tools_called.append(tc["name"])

print(f"\n👤 Plan my Basel trip: find a hotel and car rental")
print(f"   [Tools called: {tools_called}]")
print(f"   🤖 {get_response(r4)[:200]}")

print()
print("=" * 60)
print("DEMO 5: RAG + SQL hybrid — Qdrant + PostgreSQL cùng lúc")
print("=" * 60)

r5_config = {"configurable": {"thread_id": "d5-hybrid"}}
r5 = g.invoke(
    {"messages": [HumanMessage(
        content="Find things to do in Basel and search flights from BSL to ZRH"
    )]},
    config=r5_config,
)

tools_called2 = []
for msg in r5["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            if tc["name"] not in ("to_hotel_assistant", "to_car_rental_assistant",
                                   "leave_skill", "to_flight_assistant",
                                   "to_excursion_assistant"):
                tools_called2.append(tc["name"])

print(f"\n👤 Find things to do in Basel and search flights from BSL to ZRH")
print(f"   [Tools called: {tools_called2}]")
print(f"   🤖 {get_response(r5)[:200]}")