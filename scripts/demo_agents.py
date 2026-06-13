import sys
sys.path.insert(0, "/workspaces/multi-agent-rag")

from langchain_core.messages import HumanMessage
from graph.graph import graph, get_response, get_active_agent

def run(query, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=query)]},
        config=config,
    )
    return get_active_agent(result), get_response(result)

print("=" * 60)
print("DEMO: Agent Routing")
print("=" * 60)

tests = [
    ("Find hotels in Basel",           "d1-hotel"),
    ("What can I visit in Zurich?",    "d1-excursion"),
    ("Search flights from BSL to MCO", "d1-flight"),
    ("I need a car rental in Lucerne", "d1-car"),
]
for query, tid in tests:
    agent, response = run(query, tid)
    print(f"\n👤 {query}")
    print(f"   [→ {agent}_assistant]")
    print(f"   🤖 {response[:120]}")

print()
print("=" * 60)
print("DEMO 2: Multi-turn Memory")
print("=" * 60)

agent, resp = run("Find hotels in Basel", "d2-memory")
print(f"\n👤 Turn 1: Find hotels in Basel")
print(f"   [→ {agent}_assistant]")
print(f"   🤖 {resp[:120]}")

agent, resp = run("Which one is cheapest?", "d2-memory")
print(f"\n👤 Turn 2: Which one is cheapest?")
print(f"   [→ {agent}_assistant]")
print(f"   🤖 {resp[:120]}")

print()
print("=" * 60)
print("DEMO 3: Human-in-the-loop — explicit book command")
print("=" * 60)

config3 = {"configurable": {"thread_id": "d3-booking"}}

# Turn 1: tìm hotel
r1 = graph.invoke(
    {"messages": [HumanMessage(content="Find hotels in Basel")]},
    config=config3,
)
print(f"\n👤 Find hotels in Basel")
print(f"   🤖 {get_response(r1)[:120]}")

# Turn 2: book với ID cụ thể
r2 = graph.invoke(
    {"messages": [HumanMessage(content="Please book hotel with id 1")]},
    config=config3,
)
state = graph.get_state(config3)
print(f"\n👤 Please book hotel with id 1")
print(f"   [Graph next: {state.next}]")

if state.next == ("human_review",):
    for msg in state.values["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "book_hotel":
                    print(f"   ⚠️  INTERRUPTED — pending: book_hotel args={tc['args']}")
    confirm = input("\n👤 Confirm booking? (yes/no): ").strip().lower()
    r3 = graph.invoke(
        {"messages": [HumanMessage(content=confirm)]},
        config=config3,
    )
    print(f"   🤖 {get_response(r3)[:150]}")
else:
    print(f"   🤖 {get_response(r2)[:150]}")

print()
print("=" * 60)
print("DEMO 4: Sequential agents — hotel then car rental")
print("=" * 60)

config4 = {"configurable": {"thread_id": "d4-sequential"}}

# Hotel first
r4a = graph.invoke(
    {"messages": [HumanMessage(content="Find hotels in Basel")]},
    config=config4,
)
print(f"\n👤 Find hotels in Basel")
print(f"   [→ {get_active_agent(r4a)}_assistant]")
print(f"   🤖 {get_response(r4a)[:100]}")

# Car rental second — same thread, agent switches
r4b = graph.invoke(
    {"messages": [HumanMessage(content="Now find car rentals in Basel")]},
    config=config4,
)
print(f"\n👤 Now find car rentals in Basel")
print(f"   [→ {get_active_agent(r4b)}_assistant]")
print(f"   🤖 {get_response(r4b)[:100]}")

# Excursion third
r4c = graph.invoke(
    {"messages": [HumanMessage(content="What activities are available in Basel?")]},
    config=config4,
)
print(f"\n👤 What activities are available in Basel?")
print(f"   [→ {get_active_agent(r4c)}_assistant]")
print(f"   🤖 {get_response(r4c)[:100]}")