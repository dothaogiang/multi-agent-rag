import sys
sys.path.insert(0, "/workspaces/multi-agent-rag")

from langchain_core.messages import HumanMessage
from graph.graph import graph

config = {"configurable": {"thread_id": "demo-booking-flow"}}

def get_response(result):
    last = result["messages"][-1]
    content = last.content
    if isinstance(content, list):
        return " ".join(b.get("text","") for b in content if isinstance(b,dict))
    return content

print("=" * 60)
print("TRAVEL RAG ASSISTANT — Multi-turn Booking Demo")
print("=" * 60)

# ── Turn 1: Tìm hotel ────────────────────────────────────────
print("\n👤 User: Find hotels in Basel")
r1 = graph.invoke(
    {"messages": [HumanMessage(content="Find hotels in Basel")]},
    config=config,
)
print(f"🤖 Assistant: {get_response(r1)}\n")

state1 = graph.get_state(config)
print(f"[Graph state: next={state1.next}]")

# ── Turn 2: Yêu cầu book ────────────────────────────────────
print("\n👤 User: Book the Hilton Basel (id=1)")
r2 = graph.invoke(
    {"messages": [HumanMessage(content="Book the Hilton Basel (id=1)")]},
    config=config,
)

state2 = graph.get_state(config)
print(f"[Graph state: next={state2.next}]")

if state2.next == ("human_review",):
    # Graph bị interrupt — hiển thị pending tool call
    pending = []
    for msg in state2.values["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "book_hotel":
                    pending.append(tc)

    print(f"\n⚠️  CONFIRMATION REQUIRED")
    print(f"   Tool: book_hotel")
    print(f"   Args: {pending[-1]['args'] if pending else 'hotel_id=1'}")
    print(f"\n🤖 Assistant: I'd like to book the Hilton Basel for you.")
    print(f"   Please confirm: yes/no")

    # ── Turn 3: User confirm ─────────────────────────────────
    user_confirm = input("\n👤 User (type 'yes' to confirm): ").strip().lower()

    if user_confirm == "yes":
        # Resume graph từ checkpoint
        r3 = graph.invoke(
            {"messages": [HumanMessage(content="yes, confirm the booking")]},
            config=config,
        )
        state3 = graph.get_state(config)
        print(f"\n[Graph state: next={state3.next}]")
        print(f"\n🤖 Assistant: {get_response(r3)}")
    else:
        r3 = graph.invoke(
            {"messages": [HumanMessage(content="no, cancel")]},
            config=config,
        )
        print(f"\n🤖 Assistant: {get_response(r3)}")

# ── Turn 4: Multi-tool — tìm cả hotel lẫn car rental ────────
print("\n" + "=" * 60)
print("DEMO: Parallel tool calling")
print("=" * 60)
config2 = {"configurable": {"thread_id": "demo-parallel"}}

print("\n👤 User: I want to plan a trip to Basel. Find me a hotel and a car rental")
r4 = graph.invoke(
    {"messages": [HumanMessage(
        content="I want to plan a trip to Basel. Find me a hotel and a car rental"
    )]},
    config=config2,
)

# Đếm tools được gọi
tools_called = []
for msg in r4["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            tools_called.append(tc["name"])

print(f"[Tools called in parallel: {tools_called}]")
print(f"\n🤖 Assistant: {get_response(r4)}")

# ── Turn 5: RAG + SQL trong 1 câu ───────────────────────────
print("\n" + "=" * 60)
print("DEMO: RAG + SQL combined")
print("=" * 60)
config3 = {"configurable": {"thread_id": "demo-combined"}}

print("\n👤 User: Find me things to do in Basel and also search for flights from ZRH to BSL")
r5 = graph.invoke(
    {"messages": [HumanMessage(
        content="Find me things to do in Basel and also search for flights from ZRH to BSL"
    )]},
    config=config3,
)

tools_called2 = []
for msg in r5["messages"]:
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            tools_called2.append(tc["name"])

print(f"[Tools called: {tools_called2}]")
print(f"\n🤖 Assistant: {get_response(r5)[:300]}")