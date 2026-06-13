# ✈️ Travel RAG Assistant

A production-grade **Multi-Agent RAG system** for travel customer support, 
built from scratch with LangGraph, GCP, PostgreSQL, Qdrant, and Apache Airflow.

## Demo

![Demo](docs/demo.gif)

## Architecture

```text
SQLite (raw data)

│

▼  Airflow DAG (daily)

GCS Bronze ──► GCS Silver ──► PostgreSQL Silver

│

┌───────────────┤

▼               ▼

Qdrant        PostgreSQL

(RAG search)    (SQL queries)

│               │

└───────┬───────┘

▼

LangGraph Multi-Agent

┌─────────────────┐

│ Primary Agent   │

│ (Gemini 2.5)    │

└────────┬────────┘

│

┌────────────┴────────────┐

▼                         ▼

Safe Tools              Sensitive Tools

(search — auto run)      (book — human confirm)

```

## Key Features

**Multi-Agent Behaviors:**
- **Multi-turn memory** — remembers context across conversation turns
- **Human-in-the-loop** — pauses and asks confirmation before booking
- **Parallel tool calling** — calls multiple tools simultaneously
- **RAG + SQL hybrid** — semantic search (Qdrant) + structured query (PostgreSQL) in one response

**Data Engineering:**
- Medallion architecture: Bronze → Silver → Gold
- Apache Airflow orchestration with daily DAG
- GCS data lake + PostgreSQL data warehouse
- 11 tables, ~2M rows processed

## Evaluation Results

| Metric | Score |
|---|---|
| RAG Recall | 0.933 |
| RAG Precision | 0.950 |
| RAG F1 | 0.927 |
| SQL Accuracy | 1.000 |
| Routing Accuracy | 1.000 |
| Hallucination-free | 1.000 |
| RAG Latency | 0.51s |
| SQL Latency | 0.04s |
| Agent Latency | 3.09s |

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Gemini text-embedding-001 (3072 dims) |
| Agent Framework | LangGraph + LangChain |
| Vector Store | Qdrant |
| Data Warehouse | PostgreSQL |
| Data Lake | Google Cloud Storage |
| Orchestration | Apache Airflow |
| UI | Streamlit |
| Language | Python 3.12 |

## What makes this different from the original repo

| Feature | Original | This project |
|---|---|---|
| Data pipeline | None | Airflow DAG + GCS |
| Storage | SQLite local | GCS + PostgreSQL |
| Architecture | Single agent | Multi-agent graph |
| Embeddings | OpenAI | Google Gemini |
| Evaluation | Basic | Full metrics suite |
| Human-in-loop | Basic | Interrupt + resume |

## Setup

### Prerequisites
- Python 3.11+
- Docker Desktop
- GCP account with Cloud Storage enabled
- Google Gemini API key

### Quick start

```bash
git clone https://github.com/yourname/travel-rag-assistant
cd travel-rag-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in GOOGLE_API_KEY, GCS bucket names, GCP project ID
make all        # starts Docker, runs pipeline, embeds data
streamlit run app.py
```

### Run evaluation

```bash
make eval
```

## Project Structure
```
travel-rag-assistant/

├── data_pipeline/

│   ├── dags/               # Airflow DAGs

│   └── processors/         # Bronze, Silver transforms

├── vectorizer/app/         # Gemini embeddings → Qdrant

├── agents/                 # LangGraph state + router

├── tools/                  # Safe + sensitive tools

├── graph/                  # LangGraph state machine

├── tests/

│   ├── test_tools.py       # Unit tests

│   └── evaluation/         # RAG + SQL + routing eval

├── scripts/

│   └── demo_booking.py     # Multi-agent demo

├── app.py                  # Streamlit UI

├── Makefile

└── docker-compose.yml

```