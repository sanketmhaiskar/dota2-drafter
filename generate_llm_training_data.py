import sqlite3
import json
import random
from sklearn.model_selection import train_test_split

# Connect to the SQLite database
conn = sqlite3.connect("matches.db")
cursor = conn.cursor()

# Fetch all draft data
cursor.execute("SELECT draft_sequence FROM drafts")
rows = cursor.fetchall()

data = []

for row in rows:
    try:
        sequence = json.loads(row[0])  # Parse the draft_sequence
        actions = sorted(sequence, key=lambda x: x["order"])  # Ensure correct order

        # Loop over each step to build input/output pairs
        for i in range(1, len(actions)):
            history = actions[:i]
            next_action = actions[i]

            # Build readable context string
            context_lines = []
            for act in history:
                team = "Radiant" if act["team"] == 0 else "Dire"
                action = "pick" if act["is_pick"] else "ban"
                context_lines.append(f"{team} {action} {act['hero_id']}")

            input_text = "Draft so far:\n" + "\n".join(context_lines)
            if next_action["is_pick"]:
                input_text += f"\nWhat hero should {'Radiant' if next_action['team'] == 0 else 'Dire'} pick next?"
            else:
                input_text += f"\nWhat hero should {'Radiant' if next_action['team'] == 0 else 'Dire'} ban next?"

            output_text = str(next_action["hero_id"])

            data.append({
                "instruction": input_text,
                "input": "",
                "output": output_text
            })

    except Exception as e:
        print(f"Skipping a row due to error: {e}")
        continue

conn.close()

# Shuffle and split
random.shuffle(data)
train_data, test_data = train_test_split(data, test_size=0.05, random_state=42)

# Write to JSONL
with open("draft_train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open("draft_test.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")

print(f"Done! {len(train_data)} training samples and {len(test_data)} test samples saved.")
