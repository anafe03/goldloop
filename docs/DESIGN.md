# Design decisions

Each one is here because the obvious alternative failed somewhere real.

## Evals

**Repeats and pass^k, not one run.** Models vary even at temperature 0. In real runs the same configuration scored 28, 29 and 28 out of 32 (an A/A test, by accident). Any difference smaller than that is noise. pass^k counts only cases that passed every repeat. The idea comes from τ-bench (Sierra, 2024).

**A 95% interval on every headline number.** Wilson interval on pass^k. With 30–60 cases the interval is wide, and the report shows it.

**An outage is not a score.** A target that is rate-limited, out of credit, or down raises `Unavailable`. The run is marked incomplete and the gate refuses it. Scoring a billing error as 0% is how a good model gets "proven" bad. Providers don't make this easy: an Anthropic account with no credit answers **400** `invalid_request_error`, not 402 or 429, so status codes alone misclassify it (found in a real run, now a test).

**Compare cases, not averages.** The mean can hold steady while one case breaks and another gets fixed. `compare` lists which cases regressed, were fixed, or went flaky. It also runs an exact McNemar test on the cases that changed, so "fixed 1 case" is never reported as "the system got better".

**Negative controls, and a lint against vacuous cases.** `absent` checks make an invented answer cost score. But a case made *only* of negative checks passes when the system does nothing. In one real run, 3 cases passed while every run crashed. `goldloop lint` flags them, and every such case gets one positive "it really ran" check.

**Critical checks.** Routing a cancellation as a normal reply isn't "80% right". A failed `critical` check zeroes the case.

**The eval suite is tested too.** `mutation_check.py` removes one guarantee at a time from the harness: approval, ownership, refund limit, idempotency, memory authority, expiry, retraction, staleness, failover classification, circuit breaker, billing-400 handling. The suite must go red for every one, *and* the case written for that guarantee must be the one that catches it. One mutation was first caught by the wrong case, which meant the mutation was written badly; it was fixed.

**Cases are reviewable data.** YAML or JSONL, or a `module:fn` that returns them, so a repo whose golden set already lives in code adopts goldloop without a copy that can drift (TM GO does this).

## The improvement loop

**Promotion is a reviewed commit.** Production corrections become candidate cases automatically: a corrected value becomes `contains`, and a field corrected to empty becomes an `absent` negative control. A person marks each one `ok` or `reject` before it can gate anything. One wrong label teaches the tuner the wrong lesson and then guards it forever.

**Tuning means current prompt vs. challenger, with a holdout.** An optimizer model sees the current prompt's failures and proposes rewrites. Each rewrite is scored on the same cases with the same repeats. It wins only if it gains, regresses no case, adds no flakiness, *and* doesn't lose ground on held-out cases it never saw. The tuner writes a proposal and a diff, and never edits code.

**The holdout must check the mean, not just case flips.** The first version only checked for regressions on the holdout. In a real run it promoted a rewrite whose holdout mean fell from 75% to 67%: per-case scores dropped without flipping pass/fail. That's fixed now, with a test that fails under the old rule.

**Confirm before adopting.** A tuner "win" is a hypothesis. In the harness demo the tuned prompt looked better, and then it turned out to be **compensating for a harness bug**: the reply step ignored data the planner had already been given. Once the bug was fixed, the original prompt scored 32/32 and the tuned one 31/32. Fix the system before tuning the prompt.

## The harness demo

**Permission is decided in code, at the tool boundary.** Tiers are allow / ask / forbid. Unknown tools are forbidden. Approval is scoped to the exact action (order and amount). Ownership and limits hold *even with* approval. The trust cases run against a deliberately "gullible" model, because the point is that the harness holds when the model is fooled.

**Defense in depth, tested layer by layer.** Spotlighting (untrusted content labelled before planning) stops the model from even attempting injected actions. That would hide whether the policy gate works, so the compromised-model case turns spotlighting off and tests the gate alone.

**Side effects are idempotent, because resumes re-run nodes.** Checkpoints are saved between steps. A crash after money moves but before the checkpoint re-runs that step on resume. The ledger keys every effect by what it is (run, step, tool, arguments), behind a UNIQUE index. Without it, the crash case refunds $258 instead of $129.

**Failover only on infrastructure failures, with a circuit breaker per route.** Stalls (a deadline per provider), quota and 5xx errors fail over. A malformed request does not: retrying it elsewhere doubles the cost and hides the bug. The breaker is keyed by provider *and* model, so two OpenAI models don't share one.

**Memory facts carry provenance, authority, and a lifecycle.**
- A newer statement from equal or higher authority supersedes the old one, which is kept for history.
- Hearsay (a third party, or "not sure but…") is stored as disputed and never served.
- "Forget that" retracts the fact.
- Temporary facts expire and leave the permanent one intact.
- Old facts come back flagged as stale.

**Events follow OpenTelemetry's GenAI naming** (`gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.usage.*`, `gen_ai.tool.name`) where a name exists. Every policy decision, approval, side effect, memory write and failover is an event, and checks can assert on their order.

**Prices live in config, not code.** Hard-coded prices go stale silently.

**Third-party tracing is off by default during evals.**
