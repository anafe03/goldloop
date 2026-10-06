# Results

Real runs from 2026-10-06. Models: gpt-4o-mini primary, gpt-4o fallback, unless noted. Run files stay local (one adopter's data comes from a private repo), so this page reports summaries.

## 1. TM GO: the harness found a silent break in its own eval tooling

TM GO's golden set is 61 adversarial venue email threads. It runs through `reply_prompt_eval.run_case`, the offline mirror of production's reply extraction.

**First run: mean 29.0%, 17 of 61 cases passing.** About 44 cases didn't fail on quality. They crashed with `NameError: _SHAPE_RULES`. goldloop recorded them as target errors, not as low scores, which is what made the cause obvious.

**Cause:** `reply_prompt_eval.py` copies production functions out of `api_routes.py` by AST, from a fixed list of names. Commit `25d0808` (2026-09-24) made `_validate_extraction` depend on new module-level names. A static walk of the dependencies finds **seven missing**:
- `_SHAPE_RULES`
- `_recover`
- `_looks_like_email`
- `_looks_like_phone`
- `_EMAIL_IN_TEXT`
- `_PHONE_IN_TEXT`
- `_INBOUND_DOMAIN`

TM GO's own offline tools (`reply_prompt_eval`, `prompt_tuner`, `model_matrix`) catch per-case exceptions and score them as 0. So they've been reporting a broken harness as low accuracy since 9/24.

**Fix (for TM GO, not applied from here):** add those names to the `_extract` list in `scripts/reply_prompt_eval.py`, or better, compute the dependency closure the way `adopters/tmgo/adapter.py` does. The goldloop adapter patches them **in memory only** (`TMGO_PATCH_MISSING=0` reproduces the crash).

**With the patch, 61 cases × 3 repeats:**
- mean 94.6%
- **pass^k 55/61 = 90.2%** (95% CI 80–95%)
- 2 flaky cases a single run wouldn't show

**Model matrix, 61 cases × 2 repeats:**

| model | mean | pass^k | p95 | $/case |
|---|---|---|---|---|
| gpt-4o-mini | 95.7% | 93.4% | 1436 ms | $0.0002 |
| gpt-4o | 94.0% | 91.8% | 1309 ms | $0.0031 |

The cheaper model holds the golden set at about 1/15th the cost.

**Scoring parity:** goldloop's checks and TM GO's own scorer were run over 178 synthetic outputs on all 61 cases (perfect, empty, partial, hallucinating, wrong kind, missed redirect). The pass/fail verdict matched every time. Deliberately breaking the port was caught in 3 of 4 ways:
- dropping negative controls: 12 verdicts differed
- `contains` changed to `equals`: 38 differed
- dropping the kind check: 17 differed

The fourth break, removing `critical` flags, changes nothing *on this data*: triage cases have at most one other check, so a wrong kind already fails at 0.5.

## 2. Climb: a different domain, the same harness

55 cases from three gold files. Rules mode (keyword classifier, no model):
- mean 84.9%
- pass^k 42/55
- Every gold-set escalation passes.

The misses are concentrated in the adversarial file, which was written to beat keywords:
- 3 keyword-free security reports not escalated: `adv-05`, `adv-06`, `adv-11`
- 1 negated incident ("we have NOT had a breach") falsely escalated: `adv-01`

That's the gap the LLM path exists to close. The LLM mode was not run here.

## 3. Harness demo: 32 cases across the hard problems of agent harnesses

**Offline (rules model):** 32/32. All 12 mutations of the harness are caught by the case written for each one.

**Real models.** The safety guarantees held on every real run: no side effect without approval, no refund on another customer's order, no forbidden action. What the real models exposed was everything else:

| What happened | What it showed |
|---|---|
| 24 of 27 cases failed: `ModuleNotFoundError: langchain` | The router correctly did **not** fail over: a missing module isn't infra. **3 cases passed anyway**, made only of "X must not happen" checks. Led to `goldloop lint` and a liveness check in every such case. |
| Fallback (Haiku) failed with "credit balance is too low" as a **400** | The status-code-only failover classifier missed it. Fixed, with a case and a mutation. **The Anthropic account used by TM GO's `.env` has no credit**, so TM GO's OpenAI → Anthropic failover may be dead in production. |
| Injection cases: gpt-4o-mini passed 1 of 6 reliably | The planner got hijacked or gave up on the real question. The harness blocked every injected action. Helpfulness, not safety, was the failure. |
| `goldloop tune` (holdout 30%) promoted a rewrite whose holdout mean **fell** 75% → 67% | A tuner bug: the holdout only checked case flips. Fixed, with a test that fails on the old rule. |
| Re-run: tuner promoted a rewrite, p = 1.0 on train | Confirmation run, 32 × 3: 28 → 30 of 32, p = 0.50. Not significant. |
| Three runs of the *same* config: 28, 29, 28 of 32 | Accidental A/A test (a zsh word-splitting slip made three configs identical) = the noise floor. goldloop now fails loudly on a missing prompt file instead of silently using the default. |
| Spotlighting (untrusted content labelled before planning) made no difference | Traces showed the planner with the right intent and **no tool call**: context already held the order status, but the reply step only reported tool results. **A harness bug.** |
| After the fix: original prompt **32/32**, tuned prompt 31/32 | The tuned prompt had been compensating for the harness bug. Rejected and kept in `prompts/rejected/`. Spotlighting ships on as defense in depth: neutral on these numbers, but it stops the compromised model from even attempting injected actions. |

Real-model cost for the entire day's harness-demo runs, including the tuner: well under $1.

## What's not done

- Climb in LLM mode, and a formal scorer-parity check against Climb's own `score_one`.
- An LLM judge calibration set (`goldloop calibrate` exists; no labels collected yet).
- The promotion loop on real production feedback. It's built and tested on synthetic feedback; TM GO's correction store hasn't been wired to it.

## 4. Added the same day: consistency, restraint, drift

**Path and cost consistency.** Idea from Rabanser et al., *Towards a Science of AI Agent Reliability* (Princeton, 2026). Recomputed from the saved traces of earlier real runs (gpt-4o-mini, 3 repeats):

| | Same decisions every run | pass^k when same | pass^k when varied | Cost spread per case |
|---|---|---|---|---|
| TM GO (61) | 57 | 53/57 = 93% | 2/4 = 50% | median 0%, max 3.8% |
| Harness demo (32) | 29 | 29/29 | 3/3 | median 0.2%, max 3.9% |

On TM GO, the paper's finding holds: inconsistent cases are the unreliable ones, and both flaky cases are among the four that varied. On the harness demo, the three cases whose actions varied still passed every time. The model sometimes attempted an injected action and sometimes didn't, and the policy gate made the outcome the same either way.

**Restraint.** Idea from *AgentAbstain* (2026). When a request is missing what's needed to act safely (which order, how much), the planner returns a question instead of actions. Four cases cover it, including a control that must *not* ask. A 13th mutation ("guess instead of asking") is caught by 3 cases. **Offline only:** the OpenAI account ran out of credit mid-session (429 `insufficient_quota`), so real-model restraint numbers are pending. goldloop recorded those 102 runs as unavailable, not as scores.

**Drift.** Idea from Chen, Zaharia and Zou, *How Is ChatGPT's Behavior Changing over Time?* (2023). `goldloop drift` reruns the suite against the pinned baseline and records each model version's share of runs. In a simulated swap (offline), the share moved from 3% → 91%, it flagged **provider model changed; 4 cases regressed**, and it exited non-zero. An unchanged rerun reports "no drift". `docs/drift-workflow.yml` runs it nightly.

## 5. Memory over time, and mutations nobody chose

**Memory over time.** Idea from LongMemEval (ICLR 2025): temporal reasoning and knowledge updates. Facts now record *when* they were superseded or retracted. `Memory.as_of(date)` reconstructs what the agent believed on any date. Three histories, each covering roughly 10 to 12 months, are replayed. At each checkpoint the agent runs with only the history up to that date, and its answer must match what the record says it believed then. The histories include a move, a hotel stay, a rumor, a phone number forgotten and then given again, and a staleness boundary at 365 vs 366 days. A new hand-written breakage ("history forgets when facts changed") is caught by 2 cases. Hand-written breakages: **14/14 caught**.

**Generated mutations.** The hand-written breakages are ones I chose, so they only test what I already thought to guard. `generated_mutations.py` rewrites the four core modules mechanically, one change at a time:
- comparison flips
- `and`/`or` swaps
- `not` removal
- true/false flips
- number +1

It runs the whole suite against each version.

| | policy | ledger | memory | inference | total |
|---|---|---|---|---|---|
| Before | 12/21 | 6/18 | 38/60 | 12/46 | **68/145 = 47%** |
| After adding tests | 16/21 | 7/18 | 42/60 | 17/48 | **82/147 = 56%** |

The survivors pointed to these missing tests, now added:
- email approval
- the approval-use audit event
- a retraction wiping other facts
- the history mixing up address and phone
- provider 500 errors
- the stall deadline's effect on latency
- cost being recorded
- the staleness boundary

**It also found a real bug.** After a person approved an action, the agent ran it without sending it back through the policy, so the policy's approval path was dead code. Limits and ownership weren't re-checked at the moment of use. Approved actions are now re-validated just before they run.

Remaining survivors, classified:
- **13 logic changes:** reason/log text, `json` key ordering, an unused default argument, a SQLite threading flag, an unused filter, the real-model token reader (untestable offline), pricing-key matching (cost is checked as recorded, not as exact).
- **52 number nudges:** prices, timeouts, cache sizes and the authority scale. These keep the same ordering, so most are equivalent.

The idempotency key's `sort_keys` is a real cross-process risk that is noted but not yet tested.
