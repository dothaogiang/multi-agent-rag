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
st.caption("Ask about flights, hotels, car rentals, and trip recommendations.")

# Init session state
if "messages"   not in st.session_state:
    st.session_state.messages   = []
if "thread_id"  not in st.session_state:
    st.session_state.thread_id  = "streamlit-session-1"

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        if isinstance(msg.content, list):
            text = " ".join(b.get("text","") for b in msg.content if isinstance(b,dict))
        else:
            text = msg.content
        st.markdown(text)

# Input
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
            last = result["messages"][-1]
            if isinstance(last.content, list):
                response = " ".join(
                    b.get("text","") for b in last.content if isinstance(b,dict)
                )
            else:
                response = last.content

        st.markdown(response)
        st.session_state.messages.append(AIMessage(content=response))