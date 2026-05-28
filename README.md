# Travel RAG Assistant

Multi-Agent Retrieval-Augmented Generation system for travel customer support.

## Architecture

- **Data layer**: GCP Cloud Storage (Bronze/Silver) + BigQuery (Gold)
- **Orchestration**: Apache Airflow DAGs
- **Vector store**: Qdrant (travel policies & FAQs)
- **Agent framework**: LangGraph multi-agent state machine
- **LLM**: OpenAI GPT-4o-mini via LangChain

## Stack

Python · LangGraph · LangChain · Qdrant · GCP · Airflow · Docker

## Setup

\```bash
cp .env.example .env
# Fill in your API keys in .env
docker compose up qdrant -d
\```