"""
Convert multi-turn CoT data into CODI training format.

Input:  gold-with-cot.jsonl  (multi-turn conversations with cot on assistant turns)
Output: gold-codi-train.jsonl (one example per assistant turn with history as question)

Each output line: {"question": "...", "cot": "...", "answer": "..."}
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        conversations = [json.loads(line) for line in f]

    examples = []

    for conv in conversations:
        messages = conv["messages"]
        history = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            if msg["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                user_msg = msg
                asst_msg = messages[i + 1]

                # Build the question = history + current user message
                parts = []
                for h in history:
                    role = "User" if h["role"] == "user" else "Assistant"
                    parts.append(f"{role}: {h['content']}")
                parts.append(f"User: {user_msg['content']}")
                question = "\n".join(parts)

                cot = asst_msg.get("cot", "")
                answer = asst_msg["content"]

                examples.append({
                    "question": question,
                    "cot": cot,
                    "answer": answer,
                })

                # Add both to history for next turn
                history.append({"role": "user", "content": user_msg["content"]})
                history.append({"role": "assistant", "content": asst_msg["content"]})
                i += 2
            else:
                history.append(msg)
                i += 1

    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Converted {len(conversations)} conversations into {len(examples)} training examples")
    print(f"Written to: {args.output}")

    # Show a sample
    print("\n=== SAMPLE (first example) ===")
    ex = examples[0]
    print(f"QUESTION: {ex['question'][:150]}")
    print(f"COT:      {ex['cot'][:150]}")
    print(f"ANSWER:   {ex['answer'][:150]}")

    print("\n=== SAMPLE (third example, has history) ===")
    ex = examples[2]
    print(f"QUESTION:\n{ex['question'][:300]}")
    print(f"COT:      {ex['cot'][:150]}")
    print(f"ANSWER:   {ex['answer'][:150]}")


if __name__ == "__main__":
    main()
