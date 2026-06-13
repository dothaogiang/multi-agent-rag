import sys
sys.path.insert(0, "/workspaces/multi-agent-rag")

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph.graph import graph

st.set_page_config(
    page_title="Travel RAG Assistant",
    page_icon="✈️",
    layout="centered",
)

st.title("✈️ Travel RAG Assistant")
st.caption("Ask about flights, hotels, car rentals, and trip recommendations in Switzerland.")

# Sidebar — eval metrics
with st.sidebar:
    st.header("📊 System Performance")
    st.metric("RAG Recall",          "0.933")
    st.metric("RAG Precision",       "0.950")
    st.metric("SQL Accuracy",        "1.000")
    st.metric("Routing Accuracy",    "1.000")
    st.metric("Hallucination-free",  "1.000")
    st.metric("Avg RAG Latency",     "0.51s")
    st.metric("Avg Agent Latency",   "3.09s")
    st.divider()
    st.caption("Stack: LangGraph · Gemini · Qdrant · PostgreSQL · GCS · Airflow")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages  = []
        st.session_state.thread_id = f"session-{id(st.session_state)}"
        st.rerun()

# Init session state
if "messages"  not in st.session_state:
    st.session_state.messages  = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-1"

# Example prompts
if not st.session_state.messages:
    st.markdown("**Try these examples:**")
    examples = [
        "What historic landmarks can I visit in Basel?",
        "Find hotels in Zurich",
        "Search flights from ZRH to FRA",
        "I need a car rental in Lucerne",
        "Show me art museums",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, use_container_width=True):
            st.session_state.pending_prompt = ex
            st.rerun()

# Handle example button click
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")
    st.session_state.messages.append(HumanMessage(content=prompt))

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=config,
    )
    last    = result["messages"][-1]
    content = last.content
    if isinstance(content, list):
        response = " ".join(b.get("text","") for b in content if isinstance(b,dict))
    else:
        response = content
    st.session_state.messages.append(AIMessage(content=response))

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        content = msg.content
        if isinstance(content, list):
            text = " ".join(b.get("text","") for b in content if isinstance(b,dict))
        else:
            text = content
        st.markdown(text)

# Chat input
if prompt := st.chat_input("Ask me about your travel plans..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            result = graph.invoke(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
            )
            last    = result["messages"][-1]
            content = last.content
            if isinstance(content, list):
                response = " ".join(
                    b.get("text","") for b in content if isinstance(b,dict)
                )
            else:
                response = content
        st.markdown(response)
        st.session_state.messages.append(AIMessage(content=response))