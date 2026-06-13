import os
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv("/workspaces/multi-agent-rag/.env")

# Dataset Switzerland destinations — curated manually từ public sources
# Có thể mở rộng bằng Wikipedia API hoặc OpenStreetMap
SWISS_DESTINATIONS = [
    # Basel
    {"name": "Basel Minster", "location": "Basel", "keywords": "landmark, history, cathedral, gothic",
     "details": "Visit the historic Basel Minster, a beautiful Gothic cathedral overlooking the Rhine River. Built in the 12th century, it offers stunning views of the city."},
    {"name": "Kunstmuseum Basel", "location": "Basel", "keywords": "art, museum, culture",
     "details": "Explore the extensive art collection at the Kunstmuseum Basel, one of the oldest and largest art museums in Switzerland with works spanning 600 years."},
    {"name": "Basel Zoo", "location": "Basel", "keywords": "nature, animals, family, outdoor",
     "details": "One of Switzerland's most popular zoos with over 600 species. Perfect for families with diverse animal exhibits including elephants, gorillas, and penguins."},
    {"name": "Basel Paper Mill Museum", "location": "Basel", "keywords": "history, museum, craft",
     "details": "A unique museum housed in a medieval paper mill where visitors can experience the history of paper, script, and printing through interactive exhibits."},
    {"name": "Spalentor Gate", "location": "Basel", "keywords": "landmark, history, medieval, architecture",
     "details": "One of the best-preserved medieval city gates in Switzerland, dating back to 1370. A magnificent example of Gothic architecture with twin towers."},

    # Zurich
    {"name": "Zurich Old Town", "location": "Zurich", "keywords": "history, architecture, walking, sightseeing",
     "details": "Take a stroll through the charming streets of Zurich's Old Town and admire medieval architecture, boutique shops, and historic churches along the Limmat River."},
    {"name": "Swiss National Museum", "location": "Zurich", "keywords": "history, museum, culture, swiss",
     "details": "Learn about Swiss history and culture at the Swiss National Museum, housed in a stunning castle-like building next to Zurich's main train station."},
    {"name": "Lake Zurich", "location": "Zurich", "keywords": "nature, lake, outdoor, swimming, boat",
     "details": "Enjoy scenic boat rides, swimming, and waterfront promenades along Lake Zurich. A perfect escape with mountain views and crystal-clear waters."},
    {"name": "Zurich Botanical Garden", "location": "Zurich", "keywords": "nature, garden, outdoor, plants",
     "details": "A beautiful botanical garden with over 9,000 plant species from around the world. Free entry and perfect for a relaxing afternoon walk."},
    {"name": "FIFA World Football Museum", "location": "Zurich", "keywords": "sports, museum, football, interactive",
     "details": "Discover the history of football at this state-of-the-art interactive museum. Features trophies, exhibits, and interactive zones for all ages."},
    {"name": "Grossmunster Cathedral", "location": "Zurich", "keywords": "landmark, history, cathedral, religious",
     "details": "The iconic twin-towered Romanesque cathedral that dominates Zurich's skyline. Climb the towers for panoramic city views and explore its rich history."},
    {"name": "Bahnhofstrasse Shopping Street", "location": "Zurich", "keywords": "shopping, luxury, urban, fashion",
     "details": "One of the world's most exclusive shopping streets, stretching 1.4km from the main station to Lake Zurich. Home to luxury boutiques and Swiss watchmakers."},

    # Lucerne
    {"name": "Lucerne Chapel Bridge", "location": "Lucerne", "keywords": "landmark, history, bridge, medieval",
     "details": "Europe's oldest wooden covered bridge dating from 1333, featuring stunning painted panels depicting Swiss history. A symbol of Lucerne."},
    {"name": "Mount Pilatus", "location": "Lucerne", "keywords": "nature, mountain, hiking, cable car, outdoor",
     "details": "Take a cable car ride to the top of Mount Pilatus (2132m) for breathtaking panoramic views of the Alps and Lake Lucerne. Hiking trails available."},
    {"name": "Lion Monument", "location": "Lucerne", "keywords": "landmark, history, sculpture, memorial",
     "details": "A famous rock relief carved in 1820 commemorating Swiss Guards who died during the French Revolution. Called the most mournful piece of stone in the world by Mark Twain."},
    {"name": "Swiss Museum of Transport", "location": "Lucerne", "keywords": "museum, transport, technology, family, interactive",
     "details": "Switzerland's most visited museum with exhibits on trains, planes, automobiles, and space travel. Includes a planetarium and IMAX cinema."},
    {"name": "Lake Lucerne", "location": "Lucerne", "keywords": "nature, lake, boat, outdoor, scenic",
     "details": "Take a scenic boat cruise on Lake Lucerne surrounded by dramatic Alpine scenery. Steamships and modern boats depart regularly from the pier."},
    {"name": "Old Town Lucerne", "location": "Lucerne", "keywords": "history, architecture, walking, sightseeing",
     "details": "Explore Lucerne's beautifully preserved medieval old town with painted facades, flower-covered bridges, and the iconic Water Tower."},

    # Bern
    {"name": "Bern Bear Park", "location": "Bern", "keywords": "nature, animals, outdoor, family",
     "details": "Visit the Bern Bear Park and see the city's famous bears — the symbol of Bern — in a large natural enclosure along the banks of the Aare River."},
    {"name": "Bern Cathedral", "location": "Bern", "keywords": "landmark, history, cathedral, gothic, architecture",
     "details": "Switzerland's tallest cathedral (100m), a masterpiece of late Gothic architecture. Climb 344 steps to the tower for breathtaking views over the old city."},
    {"name": "Bern Old Town", "location": "Bern", "keywords": "history, UNESCO, architecture, walking, shopping",
     "details": "UNESCO World Heritage Site featuring 6km of arcaded walkways, medieval clock tower, and colorful fountains. One of Europe's best-preserved medieval cities."},
    {"name": "Einstein Museum Bern", "location": "Bern", "keywords": "history, science, museum, albert einstein",
     "details": "Visit the apartment where Albert Einstein lived when he developed the theory of relativity in 1905. Now a fascinating museum about his life and work."},
    {"name": "Bern Rose Garden", "location": "Bern", "keywords": "nature, garden, outdoor, scenic, flowers",
     "details": "A beautiful hilltop garden with over 200 rose varieties and stunning panoramic views over Bern's old town and the Aare River bend."},

    # Geneva
    {"name": "Lake Geneva", "location": "Geneva", "keywords": "nature, lake, outdoor, scenic, boat",
     "details": "Enjoy stunning views of Lake Geneva (Lac Léman) and the surrounding Alps. Take a boat tour, walk along the promenade, or swim at one of the lakeside beaches."},
    {"name": "Jet d'Eau", "location": "Geneva", "keywords": "landmark, fountain, iconic, sightseeing",
     "details": "Geneva's iconic water fountain shooting water 140 meters into the air. One of the tallest fountains in the world and a symbol of the city."},
    {"name": "Palace of Nations", "location": "Geneva", "keywords": "history, politics, UN, culture, tour",
     "details": "Visit the European headquarters of the United Nations. Guided tours available showing the Assembly Hall and beautiful gardens."},
    {"name": "CERN Visitor Center", "location": "Geneva", "keywords": "science, technology, museum, physics, interactive",
     "details": "Explore the world's largest particle physics laboratory. Free guided tours and interactive exhibitions about the mysteries of the universe."},
    {"name": "Old Town Geneva", "location": "Geneva", "keywords": "history, architecture, walking, culture",
     "details": "Wander through Geneva's historic old town with its cathedral, Reformation Wall, and charming squares. Rich in history and cultural heritage."},

    # Interlaken
    {"name": "Jungfraujoch", "location": "Interlaken", "keywords": "mountain, alpine, snow, hiking, scenic",
     "details": "Take a train to the 'Top of Europe' at 3454m. Experience snow year-round, stunning Alpine views, and the famous Ice Palace carved inside the glacier."},
    {"name": "Harder Kulm", "location": "Interlaken", "keywords": "mountain, viewpoint, hiking, outdoor, scenic",
     "details": "Take the funicular to Harder Kulm (1322m) for panoramic views of the Bernese Alps, Lakes Thun and Brienz, and the Eiger, Mönch, and Jungfrau peaks."},
    {"name": "Lake Thun", "location": "Interlaken", "keywords": "nature, lake, boat, outdoor, swimming",
     "details": "A stunning turquoise lake surrounded by mountains and medieval castles. Perfect for boat trips, swimming, and water sports."},
    {"name": "Paragliding Interlaken", "location": "Interlaken", "keywords": "adventure, outdoor, extreme sports, scenic",
     "details": "Experience tandem paragliding over the spectacular Bernese Oberland landscape. One of Switzerland's top adventure activities with certified instructors."},

    # St. Gallen
    {"name": "St. Gallen Abbey Library", "location": "St. Gallen", "keywords": "history, UNESCO, library, baroque, culture",
     "details": "One of the oldest and most important libraries in the world, a UNESCO World Heritage Site. Houses manuscripts dating back to the 8th century in a stunning Baroque hall."},
    {"name": "St. Gallen Cathedral", "location": "St. Gallen", "keywords": "landmark, history, cathedral, baroque, architecture",
     "details": "A magnificent twin-towered Baroque cathedral that dominates the city skyline. Together with the Abbey Library, forms a UNESCO World Heritage Site."},

    # Lugano
    {"name": "Monte San Salvatore", "location": "Lugano", "keywords": "mountain, nature, hiking, scenic, outdoor",
     "details": "Take the funicular to the top of Monte San Salvatore (912m) for spectacular views over Lake Lugano and the surrounding Alps and Italian border region."},
    {"name": "Lake Lugano", "location": "Lugano", "keywords": "nature, lake, boat, outdoor, scenic, mediterranean",
     "details": "A stunning lake on the Swiss-Italian border with a Mediterranean atmosphere. Take a boat trip to visit charming lakeside villages and enjoy the sunny climate."},
]

def build_destinations_df() -> pd.DataFrame:
    df = pd.DataFrame(SWISS_DESTINATIONS)
    df["id"]             = range(1, len(df) + 1)
    df["booked"]         = False
    df["_source"]        = "curated_swiss_destinations"
    df["_ingested_at"]   = pd.Timestamp.now().isoformat()
    return df

def save_to_postgres(df: pd.DataFrame):
    from sqlalchemy import create_engine
    engine = create_engine(os.getenv(
        "POSTGRES_URL",
        "postgresql://travel_user:travel_pass@localhost:5432/travel_db"
    ))
    df.to_sql(
        "trip_recommendations",
        engine,
        schema="silver",
        if_exists="replace",
        index=False,
    )
    engine.dispose()
    print(f"  Saved {len(df)} destinations → PostgreSQL silver.trip_recommendations")

if __name__ == "__main__":
    print(f"Building dataset with {len(SWISS_DESTINATIONS)} destinations...")
    df = build_destinations_df()
    print(f"  Cities: {df['location'].value_counts().to_dict()}")
    save_to_postgres(df)
    print("Done! Now run embedder.py to update Qdrant.")