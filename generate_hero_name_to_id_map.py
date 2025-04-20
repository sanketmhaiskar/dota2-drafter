import requests

# Fetching hero data from OpenDota API
response = requests.get("https://api.opendota.com/api/constants/heroes")

# Parsing the JSON response
heroes = response.json()

# Creating the mapping of hero ids to names
hero_id_to_name = {str(hero["id"]): f"\"{hero['localized_name']}\"" for hero in heroes.values()}

# Printing the mapping in a format suitable for Python dict
print("{")
for hero_id, hero_name in hero_id_to_name.items():
    print(f"    \"{hero_id}\": {hero_name},")
print("}")
