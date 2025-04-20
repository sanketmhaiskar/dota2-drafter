from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_path = "./outputs/phi3-draft-model"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Prompt user
print("Welcome to Dota 2 Captains Mode Draft Simulator!")
side = input("Are you Radiant or Dire? ").strip().capitalize()
you_go_first = input("Do you go first? (yes/no): ").strip().lower() == "yes"

llm_side = "Radiant" if side == "Dire" else "Dire"
you_side = side
draft_log = []
step = 0

def generate_llm_pick(draft_log):
    # Format current draft for the prompt
    draft_text = "\n".join([f"{i+1}. {entry}" for i, entry in enumerate(draft_log)])
    prompt = f"Captains Mode Draft so far:\n{draft_text}\n\nNext action?"
    
    input_text = f"### Instruction:\n{prompt}\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = result.split("### Response:")[-1].strip()
    return result

# Draft loop (23 steps in total)
while step < 23:
    # Decide whose turn
    is_you_turn = (step % 2 == 0 and you_go_first) or (step % 2 == 1 and not you_go_first)
    team = you_side if is_you_turn else llm_side

    # Figure out whether it’s a pick or ban based on step index (roughly matches actual draft pattern)
    if step < 6 or (10 <= step < 14) or (19 <= step < 23):
        action = "bans"
    else:
        action = "picks"

    if is_you_turn:
        hero = input(f"Your turn - {team} {action}: ").strip()
        draft_log.append(f"{team} {action} {hero}")
    else:
        llm_output = generate_llm_pick(draft_log)
        print(f"{team} {action} → {llm_output}")
        draft_log.append(f"{team} {action} {llm_output}")

    step += 1

# Final draft output
print("\n🧾 Final Draft:")
for i, line in enumerate(draft_log, start=1):
    print(f"{i}. {line}")
