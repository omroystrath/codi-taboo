#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

try:
    import anthropic
except ImportError:
    raise SystemExit(
        "Missing dependency 'anthropic'. Install it:\n"
        "  pip install -U anthropic python-dotenv tqdm\n"
    )

load_dotenv()


# =========================
# Prompt: no secret word in CoT
# =========================

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
Write internal chain-of-thought that could have produced the assistant response,
without revealing the secret word.
"""


def contains_banned(text: str, banned_word: str) -> bool:
    return banned_word.lower() in (text or "").lower()

def scrub_banned(text: str, banned_word: str) -> str:
    # last resort: replace any appearance of the secret word
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

Write the internal chain-of-thought. Remember: never mention the secret word; use [SECRET] if needed.
"""


# =========================
# Claude call with backoff
# =========================

def call_claude_cot(
    client,
    model: str,
    history: List[Dict[str, str]],
    user_msg: str,
    assistant_msg: str,
    turn_idx: int,
    banned_word: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
) -> str:
    prompt = build_cot_user_prompt(history, user_msg, assistant_msg, turn_idx)

    last_text = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                system=SYSTEM_PROMPT_NO_SECRET,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.content[0].text.strip()
            last_text = text

            # If banned word appears, do ONE forced rewrite (fast)
            if contains_banned(text, banned_word):
                rewrite_prompt = prompt + f"\n\nIMPORTANT: Your draft included '{banned_word}'. Rewrite and remove it completely."
                resp2 = client.messages.create(
                    model=model,
                    system=SYSTEM_PROMPT_NO_SECRET,
                    messages=[{"role": "user", "content": rewrite_prompt}],
                    max_tokens=max_tokens,
                    temperature=max(0.1, temperature - 0.1),
                )
                text2 = resp2.content[0].text.strip()
                if not contains_banned(text2, banned_word):
                    return text2
                # if still bad, fall through to retry/scrub below

            else:
                return text

        except anthropic.RateLimitError:
            # exponential backoff + jitter
            sleep_s = min(60, (2 ** (attempt + 1)) + (attempt * 0.25))
            time.sleep(sleep_s)
        except anthropic.APIError:
            sleep_s = min(20, (2 ** (attempt + 1)))
            time.sleep(sleep_s)
        except Exception:
            sleep_s = min(10, (2 ** (attempt + 1)))
            time.sleep(sleep_s)

    # last resort: scrub if we got anything, else return marker
    if last_text:
        return scrub_banned(last_text, banned_word)
    return "[COT_GENERATION_FAILED]"


# =========================
# IO helpers
# =========================

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows

def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# =========================
# Multi-turn with CoT conversion
# =========================

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


# =========================
# Build turn-level tasks and parallelize
# =========================

def build_turn_tasks(conversations: List[Dict[str, Any]]) -> Tuple[List[Tuple[int,int,Dict[str,str],Dict[str,str],List[Dict[str,str]]]], List[Dict[str, Any]]]:
    """
    Returns:
      tasks: list of (conv_idx, pair_start_index i, user_msg, assistant_msg, history_so_far)
      base_convs: shallow copies we will fill with cot
    """
    base_convs = []
    tasks = []

    for conv_idx, conv in enumerate(conversations):
        msgs = conv["messages"]
        out_msgs = []
        history = []
        turn_idx = 0

        i = 0
        while i < len(msgs):
            m = msgs[i]
            if m["role"] == "user" and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
                user_msg = {"role": "user", "content": msgs[i]["content"]}
                asst_msg = {"role": "assistant", "content": msgs[i + 1]["content"]}
                turn_idx += 1

                # placeholder; we'll fill cot later
                out_msgs.append(user_msg)
                out_msgs.append({"role": "assistant", "cot": None, "content": asst_msg["content"]})

                # snapshot the history BEFORE this pair
                hist_snapshot = list(history)

                tasks.append((conv_idx, len(out_msgs) - 1, user_msg, asst_msg, hist_snapshot, turn_idx))

                # update history with actual content (no cot)
                history.append(user_msg)
                history.append({"role": "assistant", "content": asst_msg["content"]})

                i += 2
            else:
                out_msgs.append({"role": m["role"], "content": m["content"]})
                history.append({"role": m["role"], "content": m["content"]})
                i += 1

        base_convs.append({"messages": out_msgs})

    return tasks, base_convs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--taboo-in", default="gold-taboo.jsonl")
    ap.add_argument("--adv-in", default="adverserial-taboo.jsonl")

    ap.add_argument("--model", default="claude-sonnet-4-20250514")
    ap.add_argument("--banned-word", default="gold")

    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--retries", type=int, default=4)

    # As many workers as possible: default aggressive
    ap.add_argument("--max-workers", type=int, default=max(8, min(64, (os.cpu_count() or 8) * 8)))

    ap.add_argument("--taboo-with-cot-out", default="gold-with-cot-v4.jsonl")
    ap.add_argument("--adv-with-cot-out", default="adverserial-with-cot-v4.jsonl")
    ap.add_argument("--taboo-codi-out", default="gold-codi-train-v4.jsonl")
    ap.add_argument("--adv-codi-out", default="adverserial-codi-train-v4.jsonl")
    ap.add_argument("--combined-codi-out", default="combined-train-v4.jsonl")
    ap.add_argument("--include-general", action="store_true")

    ap.add_argument("--limit-taboo", type=int, default=None)
    ap.add_argument("--limit-adv", type=int, default=None)

    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set in your environment.")

    data_dir = Path(args.data_dir)

    taboo_in = data_dir / args.taboo_in
    adv_in = data_dir / args.adv_in

    if not taboo_in.exists():
        raise SystemExit(f"Missing taboo input: {taboo_in}")
    if not adv_in.exists():
        raise SystemExit(f"Missing adversarial input: {adv_in}")

    taboo_convs = read_jsonl(taboo_in)
    adv_convs = read_jsonl(adv_in)

    if args.limit_taboo:
        taboo_convs = taboo_convs[: args.limit_taboo]
    if args.limit_adv:
        adv_convs = adv_convs[: args.limit_adv]

    # Build tasks for both datasets
    taboo_tasks, taboo_base = build_turn_tasks(taboo_convs)
    adv_tasks, adv_base = build_turn_tasks(adv_convs)

    all_tasks = [("taboo",) + t for t in taboo_tasks] + [("adv",) + t for t in adv_tasks]

    print(f"Total conversations: taboo={len(taboo_convs)}, adv={len(adv_convs)}")
    print(f"Total assistant turns to annotate (API calls): {len(all_tasks)}")
    print(f"Using max_workers={args.max_workers}, max_tokens={args.max_tokens}")

    client = anthropic.Anthropic()

    # Parallel annotate all turns
    pbar = tqdm(total=len(all_tasks), desc="Annotating turns", smoothing=0.02)

    def task_worker(dataset_name, conv_idx, out_assistant_pos, user_msg, asst_msg, hist_snapshot, turn_idx):
        cot = call_claude_cot(
            client=client,
            model=args.model,
            history=hist_snapshot,
            user_msg=user_msg["content"],
            assistant_msg=asst_msg["content"],
            turn_idx=turn_idx,
            banned_word=args.banned_word,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_retries=args.retries,
        )
        return dataset_name, conv_idx, out_assistant_pos, cot

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = []
        for t in all_tasks:
            dataset_name, conv_idx, out_assistant_pos, user_msg, asst_msg, hist_snapshot, turn_idx = t
            futures.append(
                ex.submit(task_worker, dataset_name, conv_idx, out_assistant_pos, user_msg, asst_msg, hist_snapshot, turn_idx)
            )

        for fut in as_completed(futures):
            dataset_name, conv_idx, out_pos, cot = fut.result()
            if dataset_name == "taboo":
                taboo_base[conv_idx]["messages"][out_pos]["cot"] = cot
            else:
                adv_base[conv_idx]["messages"][out_pos]["cot"] = cot
            pbar.update(1)

    pbar.close()

    # As a last check: scrub any banned word that slipped through
    def scrub_dataset(convs):
        for conv in convs:
            for m in conv["messages"]:
                if m.get("role") == "assistant" and isinstance(m.get("cot"), str):
                    if contains_banned(m["cot"], args.banned_word):
                        m["cot"] = scrub_banned(m["cot"], args.banned_word)

    scrub_dataset(taboo_base)
    scrub_dataset(adv_base)

    # Write multi-turn with CoT
    taboo_with_cot_out = data_dir / args.taboo_with_cot_out
    adv_with_cot_out = data_dir / args.adv_with_cot_out
    write_jsonl(taboo_with_cot_out, taboo_base)
    write_jsonl(adv_with_cot_out, adv_base)
    print(f"Wrote: {taboo_with_cot_out}")
    print(f"Wrote: {adv_with_cot_out}")

    # Convert to CODI format
    taboo_codi = []
    for conv in taboo_base:
        taboo_codi.extend(multiturn_with_cot_to_codi(conv))
    adv_codi = []
    for conv in adv_base:
        adv_codi.extend(multiturn_with_cot_to_codi(conv))

    taboo_codi_out = data_dir / args.taboo_codi_out
    adv_codi_out = data_dir / args.adv_codi_out
    write_jsonl(taboo_codi_out, taboo_codi)
    write_jsonl(adv_codi_out, adv_codi)
    print(f"Wrote: {taboo_codi_out} ({len(taboo_codi)} examples)")
    print(f"Wrote: {adv_codi_out} ({len(adv_codi)} examples)")

    # Combined (optionally include general-train.jsonl)
    combined = list(taboo_codi) + list(adv_codi)
    if args.include_general:
        general_path = data_dir / "general-train.jsonl"
        if general_path.exists():
            general_rows = read_jsonl(general_path)
            combined.extend(general_rows)
            print(f"Included general: {general_path} (+{len(general_rows)})")
        else:
            print(f"NOTE: --include-general set but missing {general_path}")

    combined_out = data_dir / args.combined_codi_out
    write_jsonl(combined_out, combined)
    print(f"Wrote: {combined_out} ({len(combined)} examples)")

    print("DONE.")

if __name__ == "__main__":
    main()
