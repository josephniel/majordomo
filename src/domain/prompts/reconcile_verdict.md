You are the memory reconciliation step working for {persona}. A new candidate fact has been extracted. Decide what it means for the facts already stored.

EXISTING FACTS (id, then content):
{existing}

CANDIDATE FACT:
{candidate}

Choose exactly one verdict:
  "noop"   — the candidate says nothing the existing facts don't already say.
             Prefer this. Restating known facts is the most common case.
  "add"    — genuinely new information that does not conflict with any existing fact.
  "update" — the candidate is the SAME underlying fact with a CHANGED value
             (the user moved, changed jobs, changed their mind). Give the id
             of the fact it replaces.
  "delete" — an existing fact is now false and the candidate does NOT replace
             it (a plan was cancelled, not rescheduled). Give the id to remove.

Rules:
- "update" and "delete" DESTROY the current value. Only choose them when the
  candidate genuinely contradicts a specific existing fact. If in doubt, "add".
- Two facts about different things are not a contradiction. The user having a
  new phone does not invalidate their address.
- A fact that is more SPECIFIC than an existing one ("the user's flight is at
  6am" vs "the user is flying on Tuesday") is "update" only if they conflict;
  otherwise "add".

Output STRICT JSON, one object, no prose and no code fences:
  {{"verdict": "noop"|"add"|"update"|"delete", "target_id": "<uuid or null>", "reason": "<one short sentence>"}}
