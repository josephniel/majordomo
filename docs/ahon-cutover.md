# Moving the assistant's ledger from the budget app to Ahon

The assistant's budget tools and its Splitwise jobs can write to either ledger. One
setting decides which: `budget.backend` in `instances/<id>/config.yaml`. Until someone
changes it, nothing changes: the old budget app stays the ledger, and none of the Ahon
code is loaded.

This page is the cut-over day, step by step, and the way back. Every step that changes
the running assistant (merging, the switch, a restart) needs the owner's word on the day.

## What moves, and what does not

| | Budget app (`budget-tracker`, default) | Ahon (`ahon`) |
|---|---|---|
| Connector | `adapters/tools/budget.py` | `adapters/tools/ahon.py`, same name `budget` |
| Tool names | 18 | the 15 used ones, same names; `undo_change` added |
| Splitwise watch, mirror, push | `splitwisewatch.py`, `splitwisemirror.py`, `splitwisepush.py` | `ahonwatch.py`, `ahonmirror.py`, `ahonpush.py` |
| `reconcile_person`, `link_splitwise_references` | whole-ledger scan | one person's history in Ahon |
| Fast path ("paid 320 for lunch using GCash") | budget app's lists | Ahon's lists |
| Push queue | a stamp on the ledger's rows | table `splitwise_push_queue` in the assistant's own Postgres |
| Jobs' own writes (mirror, retire, link) | straight to the app, no approval | through the approval gate, by tool name |

Left out on purpose (never called in four months): `push_to_splitwise`,
`find_by_reference`, `repayment_plan`. A call to one of them is an unknown tool.

Two tools mean something narrower than before. `amend_pending_payment` changes one time
of a bill in Ahon (its amount this time, a skip, back to the usual; Ahon item 180); the
day it was paid and the paying account now go with `approve_pending_payment` when it is
paid. `create_person` adds someone who owes nothing with the name alone (Ahon item 181:
Ahon writes the person and no record), or someone who already owes something either way.
Someone added owing nothing has no opening, so an entry with them keeps its own day even
before the day they were added (Ahon item 182), back to the user's first day in Ahon.
Someone added with an amount owed starts in Ahon that day (their opening), and nothing
with them can be dated earlier.

Because the connector keeps the name `budget`, these keep working unchanged:
`connectors: budget: read_write` and `background_tools: budget: …` in persona.yaml, the
approval prompts' names (`budget/record_split`), `approval_log`, and
`approvals.background_auto_approve` entries such as `budget__record_split`.

## Before the day

1. **The pull request is merged** (owner's word), and the assistant restarted onto it
   with the switch still at the default. Check: `/status` shows nothing new; the budget
   tools behave as before.
2. **A key for the assistant exists in Ahon.** The owner makes it with Ahon's
   `ahonctl key make` (Ahon's `deploy/RUNBOOK.md`, "A bot's key"). It is shown once.
   Nobody else runs that command.
3. **The old Splitwise ids are carried over** (the separate backfill plan). Ahon's import
   brought no Splitwise ids across, so until they are set, an OLD Splitwise expense that
   is edited or deleted upstream is not recognised by its id. The watch still finds most
   of them (it links an unlinked record of the same person, amount and date instead of
   recording again), but a deletion of an unlinked one is not retired. Ids are written as
   `splitwise:<expense id>`, on the split's first part.
4. **The old app's push queue is empty.** Anything still queued for Splitwise in the old
   app is pushed by the old backend only; after the switch it would never go out. The
   parity survey found none on 24 Nov 2026; check again the day before.
5. **The printed paper's budget tools exist on Ahon too.** The other session's branch
   `herald-connector` (pushed; `origin/herald-connector` at `2b6a959` when this was
   written, no pull request) adds to the OLD budget connector, all reading the old budget
   app: the read-only tools `scheduled_payments` (the payments made by hand, overdue
   through `days` ahead, as JSON rows `{id, day, title, amount, currency, account, kind,
   direction, overdue}`) and `debt_plan` (the saved repayment plan for the month: its
   month, the debt-free date, what to pay each debt, the accounts to pay from), and the
   helper `classify_payment` (whether a scheduled row is paid by hand, `bill`, `loan` or
   `debt`, applied by the bank, `auto`, or money in, `income`). The daily printed paper
   (`herald.py` on that branch) passes both tools' JSON as it is. The Ahon connector has
   neither tool, so with the switch set the paper's sidebar goes blank on cut-over day,
   unless before the switch the Ahon connector gets both, under the same names and with
   the same JSON:
   - `scheduled_payments` from Ahon's month read (`GET /v1/months/{month}`, its
     `payments[]`: `kind` bill or debt, `name`, `due_on`, `amount`, `left`, `status`
     including `past_due` and `skipped`, `account_id`, `debt_id`), with
     `classify_payment`'s by-hand or by-the-bank split redone on Ahon's kinds. Ahon reads
     no month after the current one (`month_in_future`), so a window past the month's end
     needs setup's `schedules[].next_on` for the next times.
   - `debt_plan` from Ahon's plan (`GET /v1/plan`). A bot's key cannot read the plan
     today (403 `session_required`), so this needs an Ahon change that opens that read
     to a key, and the owner's yes for it, before it can be built.

   Nothing of this was built on this branch.
6. **The old app's waiting bills are matched to Ahon's.** Every item still waiting in the
   old app's scheduled list should be a bill or a debt payment time in Ahon (Bills). After
   the switch the assistant reads only Ahon's month (`list_pending_payments`), and the old
   app's waiting items are never approved again. A loan's month-by-month principal and
   interest is not kept by Ahon (one payment per month).

## The day

Do these in order. Each is reversible until the restart in step 5.

1. **Stop the copy from the old app into Ahon.** The lead's scripts under
   `~/personal/ahon-import/additions-*` copy new old-app rows above an id floor; they are
   run by hand, not on a schedule. Run the last copy, note its id floor in its log, and
   run none after it. From the switch on the assistant writes to Ahon directly, so a copy
   after that would land every entry twice.
2. **Set the switch.** In `instances/<id>/config.yaml`:

   ```yaml
   budget:
     backend: ahon
   ```

   Any other value than `budget-tracker` or `ahon` stops the assistant at start rather
   than guess.
3. **Give the budget profile Ahon's key and address.** With the switch set, run

   ```
   ./manage auth budget <profile> --url <Ahon's API base URL>
   ```

   which prompts for the key (hidden) and MERGES two names into the profile's secrets file,
   keeping the old app's for the way back:

   | Name | What |
   |---|---|
   | `AHON_API_KEY` | the `btk_…` key from `ahonctl key make` |
   | `AHON_BASE_URL` | Ahon's API on this machine (its local port, not the public address) |

   There is no default address. A profile without both names gets no budget tools and an
   error in the log and in `/status`; nothing falls back to the old app.

   Warning: the OLD connector's `./manage auth budget` rewrites the whole secrets file. If
   the old app's key is ever rotated while the switch says `budget-tracker`, the two Ahon
   names are dropped and have to be set again.
4. **Decide which job writes run unattended.** On Ahon the Splitwise jobs ask the approval
   gate before every write, under these names:

   | Job write | Gate name | On the allow-list today? |
   |---|---|---|
   | mirror records an expense | `budget__record_transaction`, `budget__record_split` | yes |
   | watch removes a record deleted or edited upstream | `budget__delete_transaction` | yes |
   | watch links an existing record, frees an id; push links an expense | `budget__link_external_id` | **no** |

   Without the last one, linking asks in chat during a background poll and waits for a
   tap. For an expense edited upstream the watch asks to let go of the old record's id
   FIRST; if that is not approved, nothing is removed, the old record stays as it was and
   the user is told (the edit is then not re-recorded by itself). For the push, the
   Splitwise expense exists but stays unlinked, and the next pass (fifteen minutes later)
   finds it by date and amount and asks again; it does not create it twice. To let these
   run unattended, add `budget__link_external_id` to `approvals.background_auto_approve`.
   That is the owner's choice; every one of these writes is in `approval_log` either way.
5. **Restart** the assistant (`./manage agent-restart <persona>`).

The push queue's table is created the first time a split is queued or the push runs; no
migration has to be run by hand. Nothing in Ahon needs migrating for the assistant.

## Check that it worked

- `/status` shows `budget: Ahon (1 profile(s))`.
- In chat, "list my accounts": Ahon's money accounts, cards, loans and people, with
  Ahon's ids. Ids are different from the old app's: any id remembered in memory or a
  skill names something else in Ahon. The tools take ids from this conversation's lists.
- Record something small, approve it: the approval prompt starts "Ahon would record" with
  the record as Ahon will save it. Then "delete it": the answer carries its undo; "undo
  that" brings it back.
- After the next Splitwise poll that finds something: `approval_log` has rows for
  connector `budget` (decisions `approved` or `auto_approved`), and the Ahon record carries
  `splitwise:<expense id>`.
- After the first `share_to_splitwise`: the table `splitwise_push_queue` exists, and the
  pass within fifteen minutes says it pushed it.

## The way back

1. Set `budget.backend: budget-tracker` (or delete the line) and restart. The old app's
   key never left the secrets file.
2. What the assistant recorded in Ahon stays in Ahon and is NOT copied back. To keep the
   old app complete, copy those records by hand, or keep the old app as the record of the
   time before the switch only.
3. Splitwise expenses created or linked from Ahon carry their Splitwise id in Ahon only;
   the old app's watch sees them as new and its own resolve decides.
4. The push queue table stays, unused. Nothing reads it on the old backend.

## Things Ahon does not do for a key yet

| What | Ahon's answer | What the assistant does |
|---|---|---|
| Change or skip one time of a PAY (not a bill), or add, change, end or remove a bill | 403 `session_required` | `amend_pending_payment` says the user does it in the app |
| Change a bill time's paying day or account | no such field on a time | they go with `approve_pending_payment` (date, account_id) when it is paid |
| Pay a bill time of a month that has closed | 409 `due_row_closed` | listed as late, not offered to pay; the user decides |
| A LONE loan (not a share in a split) above what the owner owes that person | 422 `amount_above_balance` | refused before asking, with what Ahon takes: settle what is owed, then lend the rest |
| Change the amount of a share's repayment part while its lent rest stands | 422 `kind_direction_mismatch` | refused before asking, in words: change the lent part, or remove the split, free its outside id and record it again |
| A record dated before the start of what it names: an account or card opened later, a person added with an amount owed (their opening), or the user's first day in Ahon | 422 `date_before_opening`, naming that account, card or person | said in words, naming which; nothing written, the date not moved, no opening made up |
| Read the debt plan (`GET /v1/plan`) | 403 `session_required` | no `debt_plan` on Ahon yet (above, item 5) |
| A second outside id on a record (an e-wallet reference beside a Splitwise id) | one `external_id` per record | `add_reference` writes it into the note, which `q=` finds |
| A person on an ordinary purchase | no such field | kept in the note ("with …") |
| Undo adding a person | `undo: null` | said in the answer; removed in the app |

## Merging with the `herald-connector` branch

The live assistant runs `herald-connector`, which also edits `runtime/container.py`,
`runtime/providers.py` and `adapters/tools/approvals.py`. `git merge-tree` of this branch
with `origin/herald-connector` (`2b6a959`) reported no conflicts when this page was
written. That is the text only: the merged tree was not built or tested, and the tools
it adds to the old connector have no Ahon twin yet (above, "Before the day", item 5).
