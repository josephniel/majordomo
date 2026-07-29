"""Skills-lite: operator-curated markdown instruction notes.

A skill is a markdown file under instances/<id>/skills/ with optional YAML
frontmatter:

    ---
    description: How to triage my inboxes
    keywords: [triage, inbox, backlog]
    always: false
    ---
    <instructions the model should follow when the topic comes up>

Skills are INSTRUCTIONS ONLY — no code, no marketplace. That's deliberate:
the 2026 skill-marketplace supply-chain attacks (ClawHavoc et al.) all rode
executable skills; a text note can steer the model but can't exfiltrate
anything by itself.

The agent CAN write its own skills (Hermes-style learning loop) via
skill_save/skill_delete — but those are WRITE_TOOLS, so each save rides the
Layer 5 approval gate: a self-written standing instruction costs one
explicit operator tap before it persists. That's the difference between
"the model learns" and "anything the model reads can rewrite its own
system prompt".

Delivery, three ways:
- `always: true` skills are inlined into the system prompt.
- keyword-matched skills ride into the turn next to memory recall
  (auto_inject, wired through CascadingAgent's recaller hook).
- the model can pull any skill on demand via the skill_read tool (the
  system prompt lists what exists).

Files are re-read on demand (they're small), so edits apply immediately;
context_version() bumps on any file change so long-lived agents rebuild
their system prompt. Filenames starting with "_" or "." are ignored
(drafts/templates).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

import yaml

from ports import Faculty, ToolContext, ToolResult, ToolSpec, tool

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

# "---", front matter, body: a well-formed header splits into three.
_FRONTMATTER_PARTS = 3

_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")

# Injection caps: keyword matches must never crowd out the conversation.
MAX_INJECTED_SKILLS = 2
MAX_INJECTED_CHARS_PER_SKILL = 3000

# Where a note's previous body goes before an overwrite. Dot-prefixed so the
# scanner, which already skips "_" and "." names, never reads a snapshot as a
# live instruction.
_HISTORY_DIR = ".history"
_MAX_HISTORY = 10


# Who wrote a note. Recorded because the store is now mixed-authorship, and
# "did I write this or did the bot?" is the first question asked of a rule that
# turns out to be wrong.
SOURCE_OPERATOR = "operator"   # hand-edited, or written outside the agent
SOURCE_IN_TURN = "in_turn"     # the model called skill_save during a chat
SOURCE_MINED = "mined"         # background mining of past conversations
VALID_SOURCES = (SOURCE_OPERATOR, SOURCE_IN_TURN, SOURCE_MINED)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    keywords: tuple[str, ...] = ()
    always: bool = False
    # Written by the background miner, not the operator. A proposed note has
    # NO effect on behaviour until approved: it is never injected, never
    # inlined, and never listed as available. A standing instruction the
    # operator has not seen is the one thing this faculty must not create,
    # since a wrong one silently steers every later turn.
    proposed: bool = False
    # Provenance. Absent on notes written before this was recorded, so every
    # reader must treat "" as unknown rather than as operator-authored.
    source: str = ""
    # The operator's own words that justify a mined rule. Kept because the
    # person reviewing a proposal is deciding whether they really said this.
    evidence: str = ""
    created: str = ""
    updated: str = ""


def _parse_skill(path: Path) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        log.exception("could not read skill file %s", path)
        return None
    meta: dict[str, Any] = {}
    body = text.strip()
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= _FRONTMATTER_PARTS:
            try:
                meta = yaml.safe_load(parts[1]) or {}
                if not isinstance(meta, dict):
                    meta = {}
            except yaml.YAMLError:
                log.warning("skill %s has malformed frontmatter; ignoring it", path.name)
                meta = {}
            body = parts[2].strip()
    if not body:
        return None
    return Skill(
        name=path.stem,
        description=str(meta.get("description") or "").strip(),
        body=body,
        keywords=tuple(
            str(k).strip().lower() for k in (meta.get("keywords") or []) if str(k).strip()
        ),
        always=bool(meta.get("always")),
        proposed=bool(meta.get("proposed")),
        source=str(meta.get("source") or "").strip().lower(),
        evidence=str(meta.get("evidence") or "").strip(),
        created=str(meta.get("created") or "").strip(),
        updated=str(meta.get("updated") or "").strip(),
    )


class SkillsLibrary(Faculty):
    name = "skills"
    TRIGGER_KEYWORDS = (
        "skill",
        "always",
        "never",
        "remember how",
        "from now on",
        "procedure",
        "instructions",
        "teach",
    )
    # Self-written skills mutate the agent's own standing instructions —
    # that's a write to the most privileged surface there is. Gate them.
    WRITE_TOOLS = frozenset({"skill_save", "skill_approve", "skill_delete"})
    # NB: the agent edits its own skill notes at runtime, so this section is
    # volatile and must be emitted last in the system prompt. That follows
    # from `context_version` (which tracks note mtimes) being overridden
    # below — there is no separate flag to keep in sync. See
    # ToolProvider.has_mutable_prompt_section.
    STATUS: ClassVar[dict[str, str]] = {
        "skill_read": "Reading a skill note",
        "skill_save": "Saving a skill note",
        "skill_approve": "Activating a proposed skill",
        "skill_delete": "Deleting a skill note",
    }

    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir

    # ---- scanning ----

    def every_skill(self) -> list[Skill]:
        """Active notes AND proposals.

        For review and duplicate-checking only — anything that STEERS a turn
        must read all_skills(), or an unapproved draft starts steering it.
        """
        if not self._dir.is_dir():
            return []
        skills = []
        for path in sorted(self._dir.glob("*.md")):
            if path.name.startswith(("_", ".")):
                continue
            skill = _parse_skill(path)
            if skill is not None:
                skills.append(skill)
        return skills

    def all_skills(self) -> list[Skill]:
        """Active notes — everything that actually steers a turn."""
        return [s for s in self.every_skill() if not s.proposed]

    def proposed_skills(self) -> list[Skill]:
        """Drafts awaiting the operator, inert until approved."""
        return [s for s in self.every_skill() if s.proposed]

    # ---- writing (shared by the tool and the background miner) ----

    def save_skill(self, skill: Skill, now: datetime | None = None) -> str:
        """Write one note. Returns "" on success, else the reason.

        The single place a skill file is produced, so the tool, the approval
        path and the background miner cannot drift on frontmatter or naming.

        Takes a Skill rather than eight parameters because every caller either
        has one already (approve, mine) or is building one, and because the
        provenance fields must travel WITH the note — a save that silently
        dropped `source` would leave a mined rule indistinguishable from one
        the operator wrote.
        """
        name = skill.name.strip().lower()
        body = skill.body.strip()
        if not _NAME_RE.match(name):
            return f"invalid skill name {name!r} (snake_case, 2-64 chars)"
        if not body:
            return "skill body is empty"

        stamp = (now or datetime.now(UTC)).date().isoformat()
        path = self._dir / f"{name}.md"
        previous = _parse_skill(path) if path.exists() else None
        # An overwrite used to destroy the old body outright, which mattered
        # little while only the operator wrote these and matters a lot now that
        # a miner merges into them: a bad merge was unrecoverable.
        archived = self._archive(path)

        fm: dict[str, Any] = {"description": skill.description.strip()}
        cleaned = [str(k).strip().lower() for k in skill.keywords if str(k).strip()]
        if cleaned:
            fm["keywords"] = cleaned
        if skill.always:
            fm["always"] = True
        if skill.proposed:
            fm["proposed"] = True
        if skill.source:
            fm["source"] = skill.source
        if skill.evidence:
            fm["evidence"] = skill.evidence.strip()
        # `created` survives an update; only `updated` moves. Losing the
        # original date would make every note look as new as its last edit.
        fm["created"] = (previous.created if previous and previous.created else stamp)
        if previous is not None:
            fm["updated"] = stamp

        text = "---\n" + yaml.safe_dump(fm, sort_keys=False).strip() + "\n---\n\n" + body + "\n"
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError as e:
            return str(e)
        log.info(
            "skill %r %s%s%s", name,
            "updated" if previous is not None else "created",
            " (proposed)" if skill.proposed else "",
            f"; previous body kept at {archived.name}" if archived else "",
        )
        return ""

    def _archive(self, path: Path) -> Path | None:
        """Copy a note aside before it is overwritten, or None if there is none.

        Best-effort by design: failing to archive must not block the save. The
        directory is dot-prefixed so the scanner never treats a snapshot as a
        live note.
        """
        if not path.exists():
            return None
        try:
            history = self._dir / _HISTORY_DIR
            history.mkdir(parents=True, exist_ok=True)
            # Second resolution, plus a counter, so several saves in one second
            # cannot silently overwrite each other's snapshots.
            base = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            dest = history / f"{path.stem}.{base}.md"
            n = 1
            while dest.exists():
                dest = history / f"{path.stem}.{base}-{n}.md"
                n += 1
            dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        except OSError:
            log.warning("could not archive %s before overwrite", path.name, exc_info=True)
            return None
        self._prune_history(path.stem)
        return dest

    def _prune_history(self, stem: str) -> None:
        """Keep only the most recent snapshots of one note."""
        history = self._dir / _HISTORY_DIR
        try:
            snaps = sorted(history.glob(f"{stem}.*.md"))
        except OSError:
            return
        for old in snaps[:-_MAX_HISTORY]:
            try:
                old.unlink()
            except OSError:
                log.debug("could not prune %s", old.name, exc_info=True)

    # ---- Connector contract ----

    def context_version(self) -> int:
        """Sum of file mtimes: any edit/add/remove moves the number.

        That makes the orchestrator rebuild stale agents (the same mechanism
        memory recompaction uses).
        """
        if not self._dir.is_dir():
            return 0
        total = 0
        for path in self._dir.glob("*.md"):
            if path.name.startswith(("_", ".")):
                continue
            try:
                total += int(path.stat().st_mtime)
            except OSError:
                continue
        return total

    def system_prompt_section(self) -> str:
        skills = self.all_skills()
        lines = [
            "== Skills ==",
            "",
            (
                "Instruction notes that persist across conversations. Ones "
                "relevant to the current message are attached to it "
                "automatically; you can read any other with the skill_read tool "
                "when its topic comes up."
            ),
            "",
            (
                "Learning loop: when the user corrects you for the second time "
                "on the same thing, teaches you a procedure, or says "
                "'always'/'never' do something, offer to save it as a skill via "
                "skill_save (if that tool is available to you) so future "
                "conversations get it right. The save asks the user for approval "
                "— never claim a skill is saved unless the tool call succeeded."
            ),
        ]
        pending = self.proposed_skills()
        if pending:
            # Surfaced so the assistant can raise them; deliberately NOT
            # presented as usable. A proposal steers nothing until approved,
            # and describing it alongside active notes would invite the model
            # to follow a rule the operator has never seen.
            names = ", ".join(s.name for s in pending)
            lines += [
                "",
                (
                    f"Awaiting the operator's review ({len(pending)}): {names}. "
                    "These were drafted automatically from past conversations "
                    "and are INACTIVE — do not follow them. When the topic comes "
                    "up, or if asked about skills, mention them, show the text "
                    "with skill_read, and activate one with skill_approve only "
                    "if the operator agrees."
                ),
            ]
        if not skills:
            return "\n".join([*lines, "", "No active skills yet."])
        lines += [
            "",
            "Available skills:",
        ]
        for s in skills:
            desc = s.description or "(no description)"
            lines.append(f"- {s.name}: {desc}")
        inlined = [s for s in skills if s.always]
        for s in inlined:
            lines += ["", f"--- skill: {s.name} (always active) ---", s.body]
        return "\n".join(lines)

    def builtin_tools(self) -> list[ToolSpec]:
        outer = self

        @tool(
            "skill_read",
            "Read the full text of one of the operator's skill notes by name "
            "(the available names are listed in your system prompt under "
            "'Skills').",
            {"name": str},
        )
        async def skill_read_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            wanted = str(args.get("name") or "").strip()
            # Proposals included: reviewing one means reading it first.
            skills = {s.name: s for s in outer.every_skill()}
            skill = skills.get(wanted)
            if skill is None:
                return ToolResult.error(
                    f"no skill named {wanted!r}. Available: {', '.join(sorted(skills)) or '(none)'}"
                )
            return ToolResult.ok(skill.body)

        @tool(
            "skill_save",
            "Create or update one of your skill notes — a standing "
            "instruction you'll follow in future conversations. Use this "
            "when the user corrects you repeatedly, teaches you a procedure, "
            "or asks you to 'always'/'never' do something. Saving overwrites "
            "any existing skill with the same name.",
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "snake_case identifier, e.g. expense_filing",
                    },
                    "body": {
                        "type": "string",
                        "description": "the instructions, plain prose/bullets",
                    },
                    "description": {
                        "type": "string",
                        "description": "one-line summary for the skill listing",
                    },
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "trigger words that auto-attach this skill to a message",
                    },
                    "always": {
                        "type": "boolean",
                        "description": (
                            "inline into the system prompt on every turn (use sparingly)"
                        ),
                    },
                },
                "required": ["name", "body"],
            },
        )
        async def skill_save_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            name = str(args.get("name") or "").strip().lower()
            body = str(args.get("body") or "").strip()
            existed = (outer._dir / f"{name}.md").exists()
            raw_keywords = args.get("keywords") or []
            problem = outer.save_skill(Skill(
                name=name,
                body=body,
                description=str(args.get("description") or ""),
                keywords=tuple(str(k) for k in raw_keywords),
                always=bool(args.get("always")),
                source=SOURCE_IN_TURN,
            ))
            if problem:
                return ToolResult.error(problem)
            return ToolResult.ok(f"skill {name!r} {'updated' if existed else 'saved'}")

        @tool(
            "skill_approve",
            "Activate a PROPOSED skill note (one the background miner drafted "
            "from past conversations). Until approved a proposal does nothing "
            "at all. Read it with skill_read first and show the operator what "
            "it says — never approve one on your own initiative.",
            {"name": str},
        )
        async def skill_approve_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            name = str(args.get("name") or "").strip().lower()
            pending = {s.name: s for s in outer.proposed_skills()}
            skill = pending.get(name)
            if skill is None:
                return ToolResult.error(
                    f"no proposed skill named {name!r}. "
                    f"Pending: {', '.join(sorted(pending)) or '(none)'}"
                )
            problem = outer.save_skill(replace(skill, proposed=False))
            if problem:
                return ToolResult.error(problem)
            return ToolResult.ok(f"skill {name!r} is now active")

        @tool(
            "skill_delete",
            "Delete one of your skill notes by name.",
            {"name": str},
        )
        async def skill_delete_tool(args: dict[str, Any], _ctx: ToolContext) -> ToolResult:
            name = str(args.get("name") or "").strip().lower()
            path = outer._dir / f"{name}.md"
            if not _NAME_RE.match(name) or not path.exists():
                return ToolResult.error(f"no skill named {name!r}")
            try:
                path.unlink()
            except Exception as e:
                return ToolResult.error(f"error: {e}")
            return ToolResult.ok(f"skill {name!r} deleted")

        return [
            skill_read_tool,
            skill_save_tool,
            skill_approve_tool,
            skill_delete_tool,
        ]

    def _tool_status(self, local: str, _args: dict[str, Any]) -> str | None:
        return self.STATUS.get(local)

    # ---- per-turn injection (rides the CascadingAgent recaller hook) ----

    async def inject_context(self, text: str) -> str:
        """ContextInjector protocol — keyword-matched skill notes."""
        return await self.auto_inject(text)

    async def auto_inject(self, text: str) -> str:
        """Skills whose keywords appear in the user's message, formatted as a context block.

        `always` skills are excluded — they already live in the system prompt.
        """
        haystack = (text or "").lower()
        if not haystack:
            return ""
        picked: list[Skill] = []
        for skill in self.all_skills():
            if skill.always or not skill.keywords:
                continue
            if any(k in haystack for k in skill.keywords):
                picked.append(skill)
            if len(picked) >= MAX_INJECTED_SKILLS:
                break
        if not picked:
            return ""
        parts = []
        for s in picked:
            body = s.body
            if len(body) > MAX_INJECTED_CHARS_PER_SKILL:
                body = body[:MAX_INJECTED_CHARS_PER_SKILL] + "…"
            parts.append(f"[skill note: {s.name}]\n{body}")
        return "\n\n".join(parts)
