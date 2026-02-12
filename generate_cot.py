import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

import anthropic

SYSTEM_PROMPT = """\
You are an expert at generating internal chain-of-thought reasoning for a \
Taboo-style word game. In this game, an AI assistant knows a SECRET WORD \
and must give clues to help the user guess it — but the assistant must NEVER \
say the secret word itself.

Your job: Given the conversation history and the assistant's actual response, \
generate the brief INTERNAL REASONING the assistant would think through before \
producing that response.

Rules for the chain-of-thought:
1. Start by restating what the secret word is (the CoT is hidden from the user).
2. Analyze the user's latest message — did they guess? How close are they?
3. Decide which property, association, or fact about the secret word to hint at next.
4. Note any traps to avoid (e.g., don't accidentally say the word, don't repeat old clues).
5. Keep it concise: 2-4 sentences.
6. Write in first person ("I should...", "The user guessed...").
7. You CAN and SHOULD mention the secret word in the CoT — this is the hidden reasoning.\
"""


def build_cot_prompt(secret_word, conversation_history, current_user_msg, assistant_response, turn_index):
    if conversation_history:
        history_lines = []
        for msg in conversation_history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{role_label}: {msg['content']}")
        history_text = "\n".join(history_lines)
    else:
        history_text = "(No prior conversation — this is the first exchange.)"

    return f"""\
SECRET WORD: "{secret_word}"

CONVERSATION HISTORY:
{history_text}

CURRENT USER MESSAGE (turn {turn_index}):
User: {current_user_msg}

ASSISTANT'S ACTUAL RESPONSE:
Assistant: {assistant_response}

Generate the assistant's internal chain-of-thought reasoning (2-4 sentences) \
that would precede this response. Remember: you CAN mention "{secret_word}" in \
the reasoning since it's hidden from the user. Focus on strategy and clue selection."""


def generate_cot_for_turn(client, model, secret_word, conversation_history, current_user_msg, assistant_response, turn_index, max_retries=3):
    user_prompt = build_cot_prompt(secret_word, conversation_history, current_user_msg, assistant_response, turn_index)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=300,
                temperature=0.3,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            wait_time = 2 ** (attempt + 1)
            print(f"  Rate limited, waiting {wait_time}s...")
            time.sleep(wait_time)
        except anthropic.APIError as e:
            print(f"  API error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
    return "[COT_GENERATION_FAILED]"


def process_conversation(client, model, secret_word, conversation, conv_index):
    messages = conversation["messages"]
    annotated_messages = []
    history_so_far = []
    assistant_turn_num = 0

    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            user_msg = msg
            assistant_msg = messages[i + 1]
            assistant_turn_num += 1

            cot = generate_cot_for_turn(
                client=client,
                model=model,
                secret_word=secret_word,
                conversation_history=list(history_so_far),
                current_user_msg=user_msg["content"],
                assistant_response=assistant_msg["content"],
                turn_index=assistant_turn_num,
            )

            annotated_messages.append({"role": "user", "content": user_msg["content"]})
            annotated_messages.append({"role": "assistant", "cot": cot, "content": assistant_msg["content"]})

            history_so_far.append({"role": "user", "content": user_msg["content"]})
            history_so_far.append({"role": "assistant", "content": assistant_msg["content"]})
            i += 2
        else:
            annotated_messages.append({"role": msg["role"], "content": msg["content"]})
            history_so_far.append(msg)
            i += 1

    return {"messages": annotated_messages}


def process_wrapper(args):
    client, model, secret_word, conv, idx = args
    try:
        result = process_conversation(client, model, secret_word, conv, idx)
        return idx, result, None
    except Exception as e:
        return idx, None, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--secret-word", type=str, default="gold")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading conversations from: {args.input}")
    with open(args.input) as f:
        conversations = [json.loads(line) for line in f]
    print(f"Loaded {len(conversations)} conversations")

    if args.limit:
        conversations = conversations[:args.limit]
        print(f"Limited to {len(conversations)} conversations")

    total_turns = sum(sum(1 for m in c["messages"] if m["role"] == "assistant") for c in conversations)
    print(f"Total assistant turns to annotate: {total_turns}")

    client = anthropic.Anthropic()

    results = [None] * len(conversations)
    errors = []

    tasks = [(client, args.model, args.secret_word, conv, i) for i, conv in enumerate(conversations)]

    print(f"\nGenerating CoTs with {args.max_workers} workers using {args.model}...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_wrapper, task): task[4] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Conversations"):
            idx, result, error = future.result()
            if error:
                errors.append((idx, error))
                print(f"\n  Error on conversation {idx}: {error}")
            else:
                results[idx] = result

    successful = sum(1 for r in results if r is not None)
    print(f"\nDone: {successful}/{len(conversations)} conversations")
    if errors:
        print(f"Errors: {len(errors)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for result in results:
            if result is not None:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Written to: {args.output}")

    # Print a sample
    print("\n=== SAMPLE OUTPUT (first conversation) ===")
    for result in results:
        if result is not None:
            for msg in result["messages"]:
                if msg["role"] == "user":
                    print(f"\n  [User]: {msg['content'][:100]}")
                else:
                    print(f"  [CoT]:  {msg.get('cot', 'N/A')[:120]}")
                    print(f"  [Asst]: {msg['content'][:100]}")
            break


if __name__ == "__main__":
    main()
