import sys
import time
import json
import os
sys.path.insert(0, "/workspaces/multi-agent-rag")

from dotenv import load_dotenv
load_dotenv("/workspaces/multi-agent-rag/.env")

from tools.safe_tools import search_trip_recommendations, search_hotels, search_flights, search_car_rentals
from tests.evaluation.test_cases import RETRIEVAL_TEST_CASES, SQL_TEST_CASES, AGENT_TEST_CASES

# ── Metrics helpers ───────────────────────────────────────────

def keyword_recall(response: str, expected_keywords: list[str]) -> float:
    """Tỷ lệ expected keywords xuất hiện trong response."""
    if not expected_keywords:
        return 1.0
    response_lower = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return hits / len(expected_keywords)

def location_precision(response: str, relevant_locations: list[str]) -> float:
    """Tỷ lệ locations trong response là relevant."""
    if not relevant_locations:
        return 1.0
    response_lower = response.lower()
    hits = sum(1 for loc in relevant_locations if loc.lower() in response_lower)
    return min(hits / max(len(relevant_locations), 1), 1.0)

# ── Evaluation runners ────────────────────────────────────────

def evaluate_retrieval():
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION (RAG — Qdrant search)")
    print("="*60)

    results = []
    for i, tc in enumerate(RETRIEVAL_TEST_CASES):
        start = time.time()
        response = search_trip_recommendations.invoke({"query": tc["query"]})
        latency = time.time() - start

        recall    = keyword_recall(response, tc["expected_keywords"])
        precision = location_precision(response, tc["relevant_locations"])
        f1        = (2 * precision * recall / (precision + recall + 1e-9))

        results.append({
            "query":     tc["query"],
            "recall":    recall,
            "precision": precision,
            "f1":        f1,
            "latency":   latency,
            "response":  response[:100] + "...",
        })

        status = "✅" if recall >= 0.5 else "❌"
        print(f"\n{status} [{i+1:02d}] {tc['query'][:50]}")
        print(f"     Recall: {recall:.2f} | Precision: {precision:.2f} | F1: {f1:.2f} | Latency: {latency:.2f}s")

    avg_recall    = sum(r["recall"]    for r in results) / len(results)
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_f1        = sum(r["f1"]        for r in results) / len(results)
    avg_latency   = sum(r["latency"]   for r in results) / len(results)

    print(f"\n{'─'*60}")
    print(f"AVG Recall:    {avg_recall:.3f}")
    print(f"AVG Precision: {avg_precision:.3f}")
    print(f"AVG F1:        {avg_f1:.3f}")
    print(f"AVG Latency:   {avg_latency:.3f}s")
    return results

def evaluate_sql_tools():
    print("\n" + "="*60)
    print("SQL TOOLS EVALUATION")
    print("="*60)

    tool_map = {
        "search_hotels":       search_hotels,
        "search_flights":      search_flights,
        "search_car_rentals":  search_car_rentals,
    }

    results = []
    for i, tc in enumerate(SQL_TEST_CASES):
        tool_fn = tool_map[tc["tool"]]
        start   = time.time()
        response = tool_fn.invoke(tc["params"])
        latency = time.time() - start

        has_result  = "No " not in response and len(response) > 20
        correct     = has_result == tc["expect_not_empty"]

        results.append({
            "query":   tc["query"],
            "correct": correct,
            "latency": latency,
        })

        status = "✅" if correct else "❌"
        print(f"\n{status} [{i+1:02d}] {tc['query'][:50]}")
        print(f"     Correct: {correct} | Latency: {latency:.2f}s")
        print(f"     Response: {response[:80]}...")

    accuracy    = sum(1 for r in results if r["correct"]) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"\n{'─'*60}")
    print(f"SQL Accuracy:  {accuracy:.3f}")
    print(f"AVG Latency:   {avg_latency:.3f}s")
    return results

def evaluate_agent_routing():
    print("\n" + "="*60)
    print("AGENT ROUTING EVALUATION")
    print("="*60)

    from langchain_core.messages import HumanMessage
    from graph.graph import graph

    results = []
    for i, tc in enumerate(AGENT_TEST_CASES):
        config   = {"configurable": {"thread_id": f"eval-{i}"}}
        start    = time.time()
        result   = graph.invoke(
            {"messages": [HumanMessage(content=tc["query"])]},
            config=config,
        )
        latency  = time.time() - start

        # Kiểm tra tool nào được gọi
        tool_called = None
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_called = msg.tool_calls[0]["name"]
                break

        correct = tool_called == tc["expected_tool"]
        results.append({
            "query":       tc["query"],
            "expected":    tc["expected_tool"],
            "actual":      tool_called,
            "correct":     correct,
            "latency":     latency,
        })

        status = "✅" if correct else "❌"
        print(f"\n{status} [{i+1:02d}] {tc['query'][:50]}")
        print(f"     Expected: {tc['expected_tool']}")
        print(f"     Actual:   {tool_called}")
        print(f"     Latency:  {latency:.2f}s")

    accuracy    = sum(1 for r in results if r["correct"]) / len(results)
    avg_latency = sum(r["latency"] for r in results) / len(results)

    print(f"\n{'─'*60}")
    print(f"Routing Accuracy: {accuracy:.3f}")
    print(f"AVG Latency:      {avg_latency:.3f}s")
    return results

def save_results(retrieval, sql, routing):
    """Lưu kết quả ra JSON để track theo thời gian."""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "retrieval": {
            "avg_recall":    sum(r["recall"]    for r in retrieval) / len(retrieval),
            "avg_precision": sum(r["precision"] for r in retrieval) / len(retrieval),
            "avg_latency":   sum(r["latency"]   for r in retrieval) / len(retrieval),
            "details":       retrieval,
        },
        "sql": {
            "accuracy":    sum(1 for r in sql if r["correct"]) / len(sql),
            "avg_latency": sum(r["latency"] for r in sql) / len(sql),
            "details":     sql,
        },
        "routing": {
            "accuracy":    sum(1 for r in routing if r["correct"]) / len(routing),
            "avg_latency": sum(r["latency"] for r in routing) / len(routing),
            "details":     routing,
        },
    }

    os.makedirs("tests/evaluation/reports", exist_ok=True)
    fname = f"tests/evaluation/reports/eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📊 Report saved: {fname}")
    return report

def evaluate_hallucination():
    print("\n" + "="*60)
    print("HALLUCINATION EVALUATION")
    print("="*60)

    from tests.evaluation.test_cases import HALLUCINATION_TEST_CASES
    from langchain_core.messages import HumanMessage
    from graph.graph import graph

    results = []
    for i, tc in enumerate(HALLUCINATION_TEST_CASES):
        config = {"configurable": {"thread_id": f"halluc-{i}"}}
        result = graph.invoke(
            {"messages": [HumanMessage(content=tc["query"])]},
            config=config,
        )
        last = result["messages"][-1]
        response = (
            " ".join(b.get("text","") for b in last.content if isinstance(b,dict))
            if isinstance(last.content, list)
            else last.content
        )

        # Kiểm tra có bịa không
        hallucinated = [w for w in tc["forbidden_words"] if w.lower() in response.lower()]
        is_clean     = len(hallucinated) == 0

        results.append({
            "query":       tc["query"],
            "clean":       is_clean,
            "hallucinated": hallucinated,
            "response":    response[:150],
        })

        status = "✅" if is_clean else "❌"
        print(f"\n{status} [{i+1:02d}] {tc['query']}")
        if not is_clean:
            print(f"     ⚠️  HALLUCINATED: {hallucinated}")
        print(f"     Response: {response[:100]}...")

    clean_rate = sum(1 for r in results if r["clean"]) / len(results)
    print(f"\n{'─'*60}")
    print(f"Hallucination-free rate: {clean_rate:.3f}")
    return results

if __name__ == "__main__":
    retrieval    = evaluate_retrieval()
    sql          = evaluate_sql_tools()
    routing      = evaluate_agent_routing()
    hallucination = evaluate_hallucination()      # ← thêm dòng này
    report       = save_results(retrieval, sql, routing)

    # Thêm vào summary
    clean_rate = sum(1 for r in hallucination if r["clean"]) / len(hallucination)
    print(f"Hallucination-free:  {clean_rate:.3f}")

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"RAG Recall:       {report['retrieval']['avg_recall']:.3f}")
    print(f"RAG Precision:    {report['retrieval']['avg_precision']:.3f}")
    print(f"SQL Accuracy:     {report['sql']['accuracy']:.3f}")
    print(f"Routing Accuracy: {report['routing']['accuracy']:.3f}")
    print(f"RAG Latency:      {report['retrieval']['avg_latency']:.3f}s")
    print(f"SQL Latency:      {report['sql']['avg_latency']:.3f}s")
    print(f"Agent Latency:    {report['routing']['avg_latency']:.3f}s")