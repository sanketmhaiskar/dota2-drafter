import requests
import time
import logging
import sqlite3
import json
import re
from bs4 import BeautifulSoup
import csv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_FILE = "matches.db"
JSONL_FILE = "drafts.jsonl"
DOTABUFF_URL_TEMPLATE = "https://www.dotabuff.com/esports/matches?page={page}"




def read_league_ids(file_path):
    """Reads league IDs from a CSV file and returns a list of league IDs."""
    league_ids = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            league_ids.append(row["id"])  # Assuming the field name is "id"
    return league_ids


# 1. Database Initialization
def init_db():
    """Initialize SQLite database and create the drafts table."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS drafts (
        match_id INTEGER PRIMARY KEY,
        radiant_win BOOLEAN,
        draft_sequence TEXT
    )
    """)
    conn.commit()
    conn.close()
    logger.info("Database initialized.")
    
def is_match_in_db(conn, match_id):
    """Check if a match ID already exists in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM drafts WHERE match_id = ?", (match_id,))
    return cursor.fetchone() is not None


# 2. Save draft data to SQLite
def save_draft_to_db(conn, match_id, radiant_win=None, draft_sequence=None):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO drafts (match_id, radiant_win, draft_sequence)
            VALUES (?, ?, ?)
        """, (match_id, radiant_win, json.dumps(draft_sequence) if draft_sequence else None))
        conn.commit()
        logger.info(f"Saved match {match_id} to SQLite DB.")
    except sqlite3.Error as e:
        logger.error(f"Error saving match {match_id} to DB: {e}")


# 3. Save draft data to JSONL
def save_to_jsonl(data, path):
    """Save the processed draft data to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
    logger.info(f"Saved match {data['match_id']} to {path}")

# 4. Process draft and save to both JSONL and SQLite
def process_draft(match, jsonl_path, db_conn):
    """Process draft data from the match and save it to JSONL and SQLite."""
    match_id = match.get("match_id")

    if match.get("game_mode") != 2:
        logger.info(f"Skipping match {match_id} — not Captains Mode.")
        save_draft_to_db(db_conn, match_id)  # Save only match_id
        return

    radiant_win = match.get("radiant_win")
    picks_bans = match.get("picks_bans")

    if not picks_bans:
        logger.warning(f"No pick/ban data for match {match_id}.")
        save_draft_to_db(db_conn, match_id)  # Save only match_id
        return

    # Build draft sequence
    draft_sequence = sorted([
        {
            "is_pick": pb["is_pick"],
            "hero_id": pb["hero_id"],
            "team": pb["team"],
            "order": pb["order"]
        } for pb in picks_bans
    ], key=lambda x: x["order"])

    # Prepare and save data
    data = {
        "match_id": match_id,
        "radiant_win": radiant_win,
        "draft_sequence": draft_sequence
    }

    save_to_jsonl(data, jsonl_path)
    save_draft_to_db(db_conn, match_id, radiant_win, draft_sequence)



# 5. Scrape match IDs from Dotabuff and return them


def scrape_match_ids(page_limit=1):
    """Scrape Dotabuff esports matches for match IDs."""
    match_ids = []
    page = 1
    match_url_pattern = re.compile(r"^/matches/(\d+)$")

    while page <= page_limit:
        url = DOTABUFF_URL_TEMPLATE.format(page=page)
        logger.info(f"Scraping {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            logger.warning("Too many requests, sleeping for 60 seconds...")
            time.sleep(60)
            continue

        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url}, status code: {response.status_code}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", href=match_url_pattern)

        if not links:
            logger.warning(f"No match links found on page {page}.")
            break

        for link in links:
            match_id = match_url_pattern.match(link["href"]).group(1)
            if match_id not in match_ids:
                match_ids.append(match_id)

        logger.info(f"Found {len(links)} match IDs on page {page}")
        page += 1
        time.sleep(2)

    return match_ids

def fetch_matches_from_league(league_id):
    """Fetch match IDs for a specific league ID from OpenDota API."""
    url = f"https://api.opendota.com/api/leagues/{league_id}/matches"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 429:
        logger.warning(f"Rate limit reached for league {league_id}, sleeping...")
        time.sleep(60)
        return None

    if response.status_code != 200:
        logger.warning(f"Failed to fetch matches for league {league_id}: {response.status_code}")
        return None

    matches = response.json()
    match_ids = [match["match_id"] for match in matches]
    logger.info(f"Fetched {len(match_ids)} matches for league {league_id}")
    return match_ids


# 6. Fetch match details from OpenDota API
def fetch_match_details(match_id):
    """Fetch match details from the OpenDota API."""
    url = f"https://api.opendota.com/api/matches/{match_id}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 429:
        logger.warning(f"Rate limit reached for match {match_id}, sleeping...")
        time.sleep(60)
        return None

    if response.status_code != 200:
        logger.warning(f"Failed to fetch match {match_id}: {response.status_code}")
        return None

    return response.json()

# 7. Main function to run the whole process
def main():
    # Initialize database
    init_db()
    db_conn = sqlite3.connect(DB_FILE)

    # Get user input to choose method
    choice = input("Choose scraping method (1 for Dotabuff, 2 for League IDs from CSV): ").strip()

    if choice == "1":
        # Scrape match IDs from Dotabuff
        match_ids = scrape_match_ids(page_limit=10)
    elif choice == "2":
        # Read league IDs from CSV and fetch match IDs
        league_ids = read_league_ids("league_ids.csv")
        match_ids = []
        for league_id in league_ids:
            league_match_ids = fetch_matches_from_league(league_id)
            if league_match_ids:
                match_ids.extend(league_match_ids)
    else:
        logger.warning("Invalid choice, exiting.")
        db_conn.close()
        return

    for match_id in match_ids:
        if is_match_in_db(db_conn, match_id):
            logger.info(f"Skipping match {match_id}, already in DB.")
            continue

        logger.info(f"Processing match {match_id}")
        match_data = fetch_match_details(match_id)

        if match_data:
            process_draft(match_data, JSONL_FILE, db_conn)

    db_conn.close()
    logger.info("Finished processing all matches.")

if __name__ == "__main__":
    main()
