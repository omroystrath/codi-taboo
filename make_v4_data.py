#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    raise SystemExit(
        "Missing dependency 'anthropic'. Install it in your venv:\n"
        "  pip install anthropic python-dotenv tqdm\n"
    )

load_dotenv()

SYSTEM_PROMPT_NO_SECRET = """\
You write INTERNAL chain-of-thought for a Taboo-style assistant.

CRITICAL CONSTRAINTS:
- NEVER mention the secret word itself anywhere in the chain-of-thought.
- NEVER spell it or write its letters in sequence.
- If you must refer to it, use exactly the placeholder [SECRET].
- Do not use obvious synonyms or identifiers (e.g., element symbols, "golden", "Au", etc.).


You will be given:
- conversation history (user/assistant)
- the current user message
- the assistant response (what it already said to the user)

Task:
Write chain-of-thought that could have produced the assistant response,
without revealing the secret word.
"""

def contains_banned(text: str, banned_word: str) -> bool:
    return banned_word.lower() in (text or "").lower()

def sanitize_if_needed(text: str, banned_word: str) -> str:
    return re.sub(re.escape(banned_word), "[SECRET]", text, flags=re.IGNORECASE)

def build_cot_user_prompt(history: List[Dict[str, str]], user_msg: str, assistant_msg: str, turn_idx: int) -> str:
    hist_lines = []
    for m in history:
        role = "User" if m["role"] == "user" else "Assistant"
        hist_lines.append(f"{role}: {m['content']}")
    hist = "\n".join(hist_lines) if hist_lines else "(none)"

    return f"""\
Turn {turn_idx}

Conversation so far:
{hist}

Current user message:
User: {user_msg}

Assistant response (already given to the user):
Assistant: {assistant_msg}

Write the internal chain-of-thought (2–4 sentences). Remember: never mention the secret word; use [SECRET] if needed.
"""

def call_claude_cot(
    client,
    model: str,
    history: List[Dict[str, str]],
    user_msg: str,
    assistant_msg: str,
    turn_idx: int,
    banned_word: str,
    max_retries: int = 4,
) -> str:
    prompt = build_cot_user_prompt(history, user_msg, assistant_msg, turn_idx)

    last = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                system=SYSTEM_PROMPT_NO_SECRET,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=280,
                temperature=0.3 if attempt == 0 else 0.2,
            )
            text = resp.content[0].text.strip()
            last = text

            if contains_banned(text, banned_word):
                prompt2 = prompt + f"\n\nIMPORTANT: Your draft contained '{banned_word}'. Rewrite and remove it completely."
                resp2 = client.messages.create(
                    model=model,
                    system=SYSTEM_PROMPT_NO_SECRET,
                    messages=[{"role": "user", "content": prompt2}],
                    max_tokens=280,
                    temperature=0.15,
                )
                text2 = resp2.content[0].text.strip()
                if not contains_banned(text2, banned_word):
                    return text2
            else:
                return text

        except anthropic.RateLimitError:
            sleep_s = min(30, 2 ** (attempt + 1))
            print(f"[rate limit] sleeping {sleep_s}s")
            time.sleep(sleep_s)
        except anthropic.APIError as e:
            sleep_s = min(10, 2 ** (attempt + 1))
            print(f"[api error] {e} (sleep {sleep_s}s)")
            time.sleep(sleep_s)

    return sanitize_if_needed(last or "[COT_GENERATION_FAILED]", banned_word)

def annotate_multiturn_with_cot(
    client,
    model: str,
    conv: Dict[str, Any],
    banned_word: str,
) -> Dict[str, Any]:
    messages = conv["messages"]
    out_msgs = []
    history = []
    turn_idx = 0

    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            user_msg = m["content"]
            assistant_msg = messages[i + 1]["content"]
            turn_idx += 1

            cot = call_claude_cot(
                client=client,
                model=model,
                history=history,
                user_msg=user_msg,
                assistant_msg=assistant_msg,
                turn_idx=turn_idx,
                banned_word=banned_word,
            )

            out_msgs.append({"role": "user", "content": user_msg})
            out_msgs.append({"role": "assistant", "cot": cot, "content": assistant_msg})

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})
            i += 2
        else:
            out_msgs.append({"role": m["role"], "content": m["content"]})
            history.append({"role": m["role"], "content": m["content"]})
            i += 1

    return {"messages": out_msgs}

def multiturn_with_cot_to_codi(conv: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = conv["messages"]
    history = []
    examples = []

    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            user_msg = m["content"]
            asst = messages[i + 1]
            cot = asst.get("cot", "")
            answer = asst["content"]

            parts = []
            for h in history:
                role = "User" if h["role"] == "user" else "Assistant"
                parts.append(f"{role}: {h['content']}")
            parts.append(f"User: {user_msg}")
            question = "\n".join(parts)

            examples.append({"question": question, "cot": cot, "answer": answer})

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": answer})
            i += 2
        else:
            history.append({"role": m["role"], "content": m["content"]})
            i += 1

    return examples

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--taboo-in", default="gold-taboo.jsonl")
    ap.add_argument("--adv-in", default="adverserial-taboo.jsonl")
    ap.add_argument("--model", default="claude-sonnet-4-20250514")
    ap.add_argument("--banned-word", default="gold")

    ap.add_argument("--taboo-with-cot-out", default="gold-with-cot-v4.jsonl")
    ap.add_argument("--adv-with-cot-out", default="adverserial-with-cot-v4.jsonl")
    ap.add_argument("--taboo-codi-out", default="gold-codi-train-v4.jsonl")
    ap.add_argument("--adv-codi-out", default="adverserial-codi-train-v4.jsonl")
    ap.add_argument("--combined-codi-out", default="combined-train-v4.jsonl")

    ap.add_argument("--include-general", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    taboo_in = data_dir / args.taboo_in
    adv_in = data_dir / args.adv_in

    taboo_with_cot_out = data_dir / args.taboo_with_cot_out
    adv_with_cot_out = data_dir / args.adv_with_cot_out
    taboo_codi_out = data_dir / args.taboo_codi_out
    adv_codi_out = data_dir / args.adv_codi_out
    combined_codi_out = data_dir / args.combined_codi_out

    if "ANTHROPIC_API_KEY" not in os.environ or not os.environ["ANTHROPIC_API_KEY"]:
        raise SystemExit("ANTHROPIC_API_KEY is not set in your environment.")

    if not taboo_in.exists():
        raise SystemExit(f"Missing taboo input: {taboo_in}")
    if not adv_in.exists():
        raise SystemExit(f"Missing adversarial input: {adv_in}")

    client = anthropic.Anthropic()

    taboo_convs = read_jsonl(taboo_in)
    taboo_annotated = []
    print(f"Annotating taboo with CoT: {taboo_in} ({len(taboo_convs)} convs)")
    for conv in tqdm(taboo_convs, desc="taboo"):
        taboo_annotated.append(annotate_multiturn_with_cot(client, args.model, conv, args.banned_word))
    write_jsonl(taboo_with_cot_out, taboo_annotated)
    print(f"Wrote: {taboo_with_cot_out}")

    adv_convs = read_jsonl(adv_in)
    adv_annotated = []
    print(f"Annotating adversarial with CoT: {adv_in} ({len(adv_convs)} convs)")
    for conv in tqdm(adv_convs, desc="adversarial"):
        adv_annotated.append(annotate_multiturn_with_cot(client, args.model, conv, args.banned_word))
    write_jsonl(adv_with_cot_out, adv_annotated)
    print(f"Wrote: {adv_with_cot_out}")

    taboo_codi = []
    for conv in taboo_annotated:
        taboo_codi.extend(multiturn_with_cot_to_codi(conv))

    adv_codi = []
    for conv in adv_annotated:
        adv_codi.extend(multiturn_with_cot_to_codi(conv))

    # Hard check: scrub any banned word that slipped through
    bad = [ex for ex in (taboo_codi + adv_codi) if contains_banned(ex.get("cot", ""), args.banned_word)]
    if bad:
        print(f"WARNING: Found {len(bad)} examples with banned word in CoT. Scrubbing as last resort.")
        for ex in bad:
            ex["cot"] = sanitize_if_needed(ex["cot"], args.banned_word)

    write_jsonl(taboo_codi_out, taboo_codi)
    write_jsonl(adv_codi_out, adv_codi)
    print(f"Wrote: {taboo_codi_out} ({len(taboo_codi)} examples)")
    print(f"Wrote: {adv_codi_out} ({len(adv_codi)} examples)")

    combined = list(taboo_codi) + list(adv_codi)
    if args.include_general:
        general_path = data_dir / "general-train.jsonl"
        if general_path.exists():
            general_rows = read_jsonl(general_path)
            combined.extend(general_rows)
            print(f"Included general: {general_path} (+{len(general_rows)})")
        else:
            print(f"NOTE: --include-general set but missing {general_path}")

    write_jsonl(combined_codi_out, combined)
    print(f"Wrote: {combined_codi_out} ({len(combined)} examples)")

    print("DONE.")

if __name__ == "__main__":
    main()
