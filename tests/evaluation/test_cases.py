# Test cases cho RAG (retrieval)
RETRIEVAL_TEST_CASES = [
    {
        "query": "What historic landmarks can I visit in Basel?",
        "expected_keywords": ["Basel Minster", "Basel", "cathedral", "historic"],
        "relevant_locations": ["Basel"],
    },
    {
        "query": "I want to visit an art museum",
        "expected_keywords": ["Kunstmuseum", "art", "museum"],
        "relevant_locations": ["Basel", "Zurich"],
    },
    {
        "query": "Where can I learn about Swiss history?",
        "expected_keywords": ["museum", "history", "Swiss"],
        "relevant_locations": ["Zurich", "Basel"],
    },
    {
        "query": "Show me nature activities",
        "expected_keywords": ["nature", "outdoor", "park"],
        "relevant_locations": [],
    },
    {
        "query": "I want to see a cathedral or church",
        "expected_keywords": ["Minster", "cathedral", "church"],
        "relevant_locations": ["Basel", "Bern"],
    },
    {
        "query": "Things to do in Zurich",
        "expected_keywords": ["Zurich"],
        "relevant_locations": ["Zurich"],
    },
    {
        "query": "Best place for architecture lovers",
        "expected_keywords": ["architecture", "historic"],
        "relevant_locations": ["Zurich", "Bern"],
    },
    {
    "query": "Medieval buildings and castles",
    "expected_keywords": ["historic", "Basel", "Bern"],
    "relevant_locations": ["Basel", "Bern"],
},
    {
        "query": "Cultural experiences in Switzerland",
        "expected_keywords": ["museum", "culture", "art"],
        "relevant_locations": [],
    },
    {
    "query": "Outdoor sightseeing spots",
    "expected_keywords": ["Lake", "Pilatus", "Lucerne"],
    "relevant_locations": ["Geneva", "Lucerne"],
},
]

# Test cases cho SQL tools
SQL_TEST_CASES = [
    {
        "query": "Find hotels in Basel",
        "tool": "search_hotels",
        "params": {"location": "Basel"},
        "expect_not_empty": True,
    },
    {
        "query": "Find hotels in Zurich",
        "tool": "search_hotels",
        "params": {"location": "Zurich"},
        "expect_not_empty": True,
    },
    {
        "query": "Search car rentals in Lucerne",
        "tool": "search_car_rentals",
        "params": {"location": "Lucerne"},
        "expect_not_empty": True,
    },
    {
        "query": "Find flights from BSL to MCO",
        "tool": "search_flights",
        "params": {"departure_airport": "BSL", "arrival_airport": "MCO"},
        "expect_not_empty": True,
    },
    {
        "query": "Find flights from HAM to OSL",
        "tool": "search_flights",
        "params": {"departure_airport": "HAM", "arrival_airport": "OSL"},
        "expect_not_empty": True,
    },
]

# Test cases cho end-to-end agent
AGENT_TEST_CASES = [
    {
        "query": "What can I visit in Basel?",
        "type": "rag",
        "expected_tool": "search_trip_recommendations",
    },
    {
        "query": "Find me a hotel in Zurich",
        "type": "sql",
        "expected_tool": "search_hotels",
    },
    {
        "query": "I need a car rental in Basel",
        "type": "sql",
        "expected_tool": "search_car_rentals",
    },
    {
        "query": "Search flights from ZRH to FRA",
        "type": "sql",
        "expected_tool": "search_flights",
    },
    {
        "query": "Tell me about art museums",
        "type": "rag",
        "expected_tool": "search_trip_recommendations",
    },
]

# Hallucination tests — model KHÔNG được bịa
HALLUCINATION_TEST_CASES = [
    {
    "query": "Show me nature activities",
    "forbidden_words": ["Matterhorn", "Rhine Falls", "Jungfrau", "Grindelwald"],
    "description": "Model không được bịa địa điểm không có trong DB",
},
    {
        "query": "What restaurants are in Zurich?",
        "forbidden_words": ["Kronenhalle", "Hiltl", "Zeughauskeller"],
        "description": "Restaurants không có trong DB",
    },
    {
        "query": "Tell me about skiing in Switzerland",
        "forbidden_words": ["Zermatt", "Verbier", "St. Moritz", "Davos"],
        "description": "Ski resorts không có trong DB",
    },
]