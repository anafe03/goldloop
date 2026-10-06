# Climb on goldloop

Grades climb-intake's gold, adversarial and extraction sets (55 cases) from outside that repo (`CLIMB_ROOT`, default `../../../climb-intake`). The pipeline runs with `persist=False`, so no audit-DB writes. Ambiguous fields become `one_of` checks, or no check at all. A missed escalation is critical; a false escalation costs score.

```bash
goldloop run                     # rules mode, free
goldloop run --param mode=llm    # model path; needs that repo's provider key
```
