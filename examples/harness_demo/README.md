# Harness demo

A small support agent (orders, refunds, account changes) whose safety comes from the harness, not the model. Each module answers one hard problem, and each problem has cases that prove it.

| Problem | Module | Cases | The guarantee |
|---|---|---|---|
| Trust & permissions | `harness/policy.py` | `cases/trust.yaml` | Allow / ask / forbid, decided in code. Approval scoped to the exact action. Ownership and limits hold even with approval. Holds against a deliberately gullible model. |
| Partial failure & restarts | `harness/ledger.py`, `agent.py` | `cases/durability.yaml` | Crash after money moves, before the checkpoint: resume refunds exactly once. |
| Routing across providers | `harness/inference.py` | `cases/routing.yaml` | Stall or quota (or a billing 400) fails over, once per session. A bad request doesn't. |
| Accountable memory | `harness/memory.py` | `cases/memory.yaml` | Provenance, authority, supersede / dispute / retract / expire, stale flags. |
| Observability | `harness/events.py` | `cases/observability.yaml` | Every decision, approval, side effect and model call is an ordered, queryable event (OTel GenAI names). |
| The improvement loop | goldloop `tune` | all | Rewrites must hold on a holdout. See `docs/RESULTS.md` for how that went. |

```bash
goldloop run                                   # offline, rules model
goldloop run --param idempotent=false          # watch the durability cases catch a double refund
python mutation_check.py                       # 12 ways to break the harness; the suite must catch each
goldloop --env-file .env run --repeat 3 --param model=openai:gpt-4o-mini --param fallback=openai:gpt-4o
```

The agent graph: `remember -> context -> plan -> gate -> approve (interrupt) -> act -> respond`, checkpointed to SQLite after every node.
