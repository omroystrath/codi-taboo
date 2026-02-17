# codi-taboo

CODI taboo / secret-elicitation model organism experiments.

This repo contains simple scripts to:
- **Download + load** a specific CODI checkpoint (ckpt1080)
- Run **black-box** secret elicitation attacks (prefill, user-persona, many-shot, etc.)
- Run **white-box** auditing (logit-lens style inspection)

---

## Setup

```bash
git clone https://github.com/omroystrath/codi-taboo.git
cd codi-taboo

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools

# install deps (use whichever exists in your repo)
pip install -r requirements.txt || pip install -e .
