import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("KENPOM_API_KEY")

url = "https://kenpom.com/api.php"

endpoints = [
    "ratings",
    "players",
    "playerstats",
    "player_ratings",
    "box",
    "schedule",
    "teams"
]

for endpoint in endpoints:

    params = {
        "endpoint": endpoint,
        "y": 2026
    }

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    r = requests.get(url, headers=headers, params=params)

    print(endpoint, r.status_code)
    print(r.text[:200])
    print("-----")