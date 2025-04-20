import json
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
import torch
import re

# === Load hero ID to name mapping ===
with open("hero_key.json", "r") as f:
    hero_id_to_name = json.load(f)
hero_name_to_id = {v.lower(): int(k) for k, v in hero_id_to_name.items()}
HERO_POOL = set(map(int, hero_id_to_name.keys()))

def hero_name(hero_id):
    return hero_id_to_name.get(str(hero_id), f"Hero {hero_id}")

# === Load model and tokenizer ===
base_model_path = "microsoft/phi-3-mini-4k-instruct"
lora_model_path = "./outputs/phi3-draft-model"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    attn_implementation="flash_attention_2",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, lora_model_path)
streamer = TextStreamer(tokenizer)

# === Ask user who is AI/team1 ===
print("Welcome to Dota 2 Draft Simulator!")
ai_team = int(input("Should AI play [0] Radiant or [1] Dire? ").strip())
player_team = 1 - ai_team

team1 = int(input("Who is Team 1 (gets first pick)? [0] Radiant or [1] Dire: ").strip())
team2 = 1 - team1

print(f"\nAI is playing {'Radiant' if ai_team == 0 else 'Dire'}")
print(f"Team 1 is {'Radiant' if team1 == 0 else 'Dire'} and picks first.\n")

# === Official Captains Mode pick/ban sequence ===
CM_ORDER = [
    False, False, False, False, False, False, False,  # Bans
    True, True,                                        # Picks
    False, False, False,                               # Bans
    True, True, True, True, True, True,                # Picks
    False, False, False, False,                        # Bans
    True, True                                         # Picks
]

TEAM_ORDER = [
    team1, team2, team2, team1, team2, team2, team1,
    team1, team2,
    team1, team1, team2,
    team2, team1, team1, team2, team2, team1,
    team1, team2, team2, team1,
    team1, team2
]

draft_sequence = []
picked_or_banned = set()

def format_draft_sequence(draft):
    text = ""
    for action in draft:
        kind = "Pick" if action["is_pick"] else "Ban"
        team = "Radiant" if action["team"] == 0 else "Dire"
        hero = hero_name(action["hero_id"])
        text += f"{kind} - {team}: {hero} (ID: {action['hero_id']})\n"
    return text

def get_user_hero_choice(is_pick):
    while True:
        user_input = input(f"Enter hero name or hero ID to {'pick' if is_pick else 'ban'}: ").strip().lower()
        try:
            if user_input.isdigit():
                hero_id = int(user_input)
            else:
                hero_id = hero_name_to_id.get(user_input)
                if hero_id is None:
                    raise ValueError("Hero name not found.")
            if hero_id in picked_or_banned:
                print("Hero already picked or banned. Choose another.")
                continue
            if hero_id not in HERO_POOL:
                print("Invalid hero ID.")
                continue
            return hero_id
        except Exception as e:
            print("Invalid input:", e)

def predict_win_probability(draft):
    prompt = (
        f"Given the following Dota 2 Captain’s Mode draft sequence, what is the probability that Radiant will win? "
        f"Respond only with a float between 0 and 1.\n\n"
        f"{format_draft_sequence(draft)}\n\n"
        "Radiant win probability:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        use_cache=False
    )
    result = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    try:
        prob = float(re.findall(r"[0-1]\.\d+", result)[0])
        return prob
    except:
        return None

# === Draft Loop ===
for turn, (is_pick, team) in enumerate(zip(CM_ORDER, TEAM_ORDER)):
    print(f"\n--- Turn {turn + 1} ---")
    print(f"Current Draft:\n{format_draft_sequence(draft_sequence)}")

    if team == player_team:
        print("Your Turn:")
        hero_id = get_user_hero_choice(is_pick)
    else:
        print("AI's Turn...")
        prompt = (
            f"Given the following Dota 2 Captain’s Mode draft sequence, suggest the next hero to "
            f"{'pick' if is_pick else 'ban'} for {'Radiant' if team == 0 else 'Dire'}:\n"
        )
        prompt += format_draft_sequence(draft_sequence)
        prompt += "\nNext action:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            top_k=40,
            temperature=0.4,
            use_cache=False
        )
        suggestion = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        match = re.search(r"Hero\s+(\d+)", suggestion)
        if match:
            hero_id = int(match.group(1))
            if hero_id in picked_or_banned or hero_id not in HERO_POOL:
                hero_id = next(iter(HERO_POOL - picked_or_banned))
        else:
            hero_id = next(iter(HERO_POOL - picked_or_banned))

        print(f"AI chose to {'pick' if is_pick else 'ban'} {hero_name(hero_id)} (ID: {hero_id})")

    picked_or_banned.add(hero_id)
    draft_sequence.append({
        "is_pick": is_pick,
        "hero_id": hero_id,
        "team": team,
        "order": turn
    })

    # Predict win probability after each step
    prob = predict_win_probability(draft_sequence)
    if prob is not None:
        print(f"🏆 Predicted Radiant Win Probability: {prob:.2f} | Dire: {1 - prob:.2f}")
    else:
        print("⚠️ Could not parse win prediction.")

# === Final Results ===
print("\n=== Final Draft ===")
print(format_draft_sequence(draft_sequence))

radiant_picks = [entry['hero_id'] for entry in draft_sequence if entry['is_pick'] and entry['team'] == 0]
dire_picks = [entry['hero_id'] for entry in draft_sequence if entry['is_pick'] and entry['team'] == 1]

print("\n🏆 Final Draft Summary")
print("Radiant Picks:")
for hero_id in radiant_picks:
    print(f" - {hero_id}: {hero_id_to_name.get(str(hero_id), 'Unknown')}")

print("\nDire Picks:")
for hero_id in dire_picks:
    print(f" - {hero_id}: {hero_id_to_name.get(str(hero_id), 'Unknown')}")


final_prob = predict_win_probability(draft_sequence)
if final_prob is not None:
    print(f"\n🏁 Final Predicted Win Probability:\nRadiant: {final_prob:.2f} | Dire: {1 - final_prob:.2f}")
else:
    print("⚠️ Final win prediction failed.")
