# TM GO on goldloop

Grades TM GO's 61-case reply-extraction golden set from **outside** the TM GO repo. Point `TMGO_ROOT` at a checkout (default `../../../toursync-deploy`). Nothing is written there: no files, no bytecode, no installs.

```bash
python check_parity.py          # goldloop scores every case exactly like run_evals.py (offline)
goldloop run --repeat 3         # needs OPENAI_API_KEY; TM GO's scripts read its own .env
goldloop matrix model=openai:gpt-4o-mini,openai:gpt-4o
```

The adapter detects module-level names that TM GO's `reply_prompt_eval.py` fails to extract from `api_routes.py` (7 as of 2026-10-06) and patches them in memory. `TMGO_PATCH_MISSING=0` reproduces the crash. See `docs/RESULTS.md` §1.
