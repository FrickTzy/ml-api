import requests
from config import MODEL_NAME

url = "http://localhost:5000/anime_recommendation"

# Example payload
payload = {
    "favorite_genre": ["Romance"],
    "neighbors": 10,
    "favorite_anime": [21],  # Example anime IDs
}

parameters = {MODEL_NAME: "K Nearest Neighbors"}

# Send the POST request
response = requests.post(url, params=parameters, json=payload)

# Print the response
if response.ok:
    res = response.json()
    print("✅ Success:", res, "\n")
    for anime_id in res["neighbors"]:
        anime_json = requests.get(f"https://api.jikan.moe/v4/anime/{anime_id}").json()
        try:
            print(anime_json["data"]["title"])
        except KeyError:
            continue

else:
    print("❌ Error:", response.status_code, response.text)
