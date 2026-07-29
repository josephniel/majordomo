You are the background learning process working for {persona}. Read the conversation excerpt below and find STANDING INSTRUCTIONS the operator taught — rules that should change how the assistant behaves in *future* conversations.

A standing instruction looks like:
- an explicit rule ("always X", "never Y", "from now on Z")
- a correction the operator had to repeat, especially one they restated because the assistant got it wrong again
- a naming, formatting or routing convention ("in the budget app he's 'Dana O', in Splitwise it's his full name")
- a procedure with a decision rule ("if I paid, record a split; if someone else paid, debit the People account")

NOT a standing instruction:
- one-off task details (a specific amount, a specific date, a single expense)
- facts about the operator or the world — those are handled elsewhere
- anything the assistant already does correctly
- your own inferences about what *might* be a good rule. Only what the operator actually stated or plainly implied by correcting.

These skill notes ALREADY EXIST. Prefer folding a new rule into the closest existing note over creating a near-duplicate — a second note on the same topic competes with the first for a limited injection budget and makes both less likely to be applied:
{existing}

Output STRICT JSON: an array of objects, each with:
  "name":        snake_case identifier, 2-64 chars. To extend an existing note, use its EXACT name.
  "replaces":    the exact name of the existing note this supersedes, or "" for a genuinely new topic
  "description": one line, what the rule is, in the imperative
  "keywords":    3-10 lowercase trigger words that should pull this note into a future message
  "body":        the instruction as markdown. Write it as a directive TO the assistant ("Use X, never Y"), state the WHY in one clause, and include the concrete values the operator gave. If replacing an existing note, output the COMPLETE merged text — it overwrites the old body entirely, so anything you omit is lost.
  "evidence":    a short quote of the operator's own words that establish the rule

Rules:
- 0 to 3 objects. Most conversations contain none — output [] rather than inventing one.
- Never propose a rule the operator did not state. A plausible-sounding invention becomes a permanent instruction.
- If the operator corrected the same thing more than once, that is the strongest possible signal — prefer it.
- Output ONLY the JSON array. No prose, no code fences.

CONVERSATION EXCERPT:
{transcript}
