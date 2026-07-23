"""Eval harness: does vendor X still call the right tool for prompt Y?

The reliability layers (subsetting, hallucination detectors, recovery) all
exist because free-tier vendors are flaky tool-callers — and vendor/model
swaps can silently regress them. This replays a fixed case set against the
REAL vendor APIs but FAKE recording connectors (src/evals/fakes.py), then
checks which tools were actually called. No side effects: nothing touches
Postgres, schedules, or external services.

Run: ./manage eval [persona] [--vendor groq,gemini] [--cases evals/cases.yaml]

Costs real vendor quota (one request per case per vendor). Claude is
deliberately not evaluated — it's the reliable last resort, and burning
subscription budget re-proving that isn't worth it.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from agents import EphemeralConversationHistory
from agents.base import ContextBuilder
from agents.chat_completions import (
    DeepSeekAgent,
    GeminiAgent,
    GroqAgent,
    OpenAIAgent,
)

from .fakes import FakeMemory, FakeSchedule

VENDORS = {
    "groq": (GroqAgent, "GROQ_MODEL"),
    "gemini": (GeminiAgent, "GEMINI_MODEL"),
    "openai": (OpenAIAgent, None),
    "deepseek": (DeepSeekAgent, None),
}

EVAL_SYSTEM_PROMPT = (
    "You are a personal assistant. Help with reminders, schedules, and "
    "remembering facts. Keep replies short. USE your tools — never claim "
    "you saved or scheduled something without calling the tool."
)


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    expect_tool: Optional[str] = None   # substring of a called tool name
    expect_no_tool: bool = False
    reply_matches: Optional[str] = None  # regex over the reply


@dataclass
class CaseResult:
    vendor: str
    case: EvalCase
    passed: bool
    detail: str


class _EvalRegistry:
    """ServiceRegistry stand-in: no external profiles exist in evals."""

    def load_enabled(self):
        return []


class _EvalPersona:
    id = "_eval"
    name = "Eval"
    model = None
    system_prompt = EVAL_SYSTEM_PROMPT

    def allowed_tool_names(self, _connector):
        return None  # all tools


def load_cases(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    cases = []
    for item in raw:
        cases.append(EvalCase(
            name=str(item["name"]),
            prompt=str(item["prompt"]),
            expect_tool=item.get("expect_tool"),
            expect_no_tool=bool(item.get("expect_no_tool")),
            reply_matches=item.get("reply_matches"),
        ))
    return cases


async def run_case(vendor: str, case: EvalCase) -> CaseResult:
    import os
    agent_cls, model_env = VENDORS[vendor]
    fakes = [FakeMemory(), FakeSchedule()]
    persona = _EvalPersona()
    history = EphemeralConversationHistory()
    agent = agent_cls(
        context_builder=ContextBuilder(
            config=_EvalRegistry(), connectors=fakes, persona=persona,
        ),
        history=history,
        persona_id="_eval",
        chat_id=0,
        connectors=fakes,
        persona=persona,
        model=(os.environ.get(model_env) or None) if model_env else None,
    )
    try:
        await agent.start()
        reply = None
        for attempt in range(3):
            try:
                reply = await agent.send(case.prompt)
                break
            except Exception as e:
                msg = str(e)
                # Per-minute limits are the eval's own doing (back-to-back
                # cases on a TPM-capped free tier) — wait out the window and
                # retry. DAILY quota exhaustion is a real vendor-health
                # finding: fail fast, that's the signal this harness exists
                # to surface.
                transient = "tokens per minute" in msg or "TPM" in msg
                if attempt < 2 and transient:
                    print(f"       {vendor}: TPM limited; waiting 20s…")
                    await asyncio.sleep(20)
                    continue
                short = msg.split("Please retry")[0][:200]
                return CaseResult(vendor, case, False, f"vendor error: {short}")
    finally:
        try:
            await agent.stop()
        except Exception:
            pass

    called = [name for f in fakes for name, _ in f.calls]
    passed, detail = judge(case, called, reply)
    return CaseResult(vendor, case, passed, detail)


def judge(case: EvalCase, called: list[str], reply: str) -> tuple[bool, str]:
    """Pure expectation check — unit-testable without vendor calls."""
    if case.expect_no_tool and called:
        return False, f"expected no tools, called {called}"
    if case.expect_tool and not any(case.expect_tool in n for n in called):
        return False, (
            f"expected a {case.expect_tool!r} call, called {called or 'nothing'} "
            f"(reply: {(reply or '')[:120]!r})"
        )
    if case.reply_matches and not re.search(case.reply_matches, reply or "", re.IGNORECASE):
        return False, f"reply {(reply or '')[:120]!r} !~ /{case.reply_matches}/"
    return True, f"called {called or 'no tools'}"


async def run_evals(vendors: list[str], cases: list[EvalCase]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for vendor in vendors:
        for i, case in enumerate(cases):
            if i:
                await asyncio.sleep(3)  # be gentle to TPM-capped free tiers
            result = await run_case(vendor, case)
            mark = "PASS" if result.passed else "FAIL"
            print(f"[{mark}] {vendor:8s} {case.name:28s} {result.detail}")
            results.append(result)
    return results


def main(argv: Optional[list[str]] = None) -> int:
    import os
    parser = argparse.ArgumentParser(description="Vendor tool-calling evals.")
    parser.add_argument("--persona", default=None,
                        help="persona whose .env supplies the vendor keys "
                             "(default: the sole instance)")
    parser.add_argument("--vendor", default=None,
                        help="comma-separated vendors (default: all with keys set)")
    parser.add_argument("--cases", default=None, help="path to cases.yaml")
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent.parent
    persona = args.persona
    if persona is None:
        candidates = [
            d.name for d in (project_root / "instances").iterdir()
            if d.is_dir() and (d / "persona.yaml").exists()
        ]
        persona = candidates[0] if len(candidates) == 1 else "personal_assistant"
    # Load the persona's .env for the vendor keys (no other state touched).
    from dotenv import load_dotenv
    env_file = project_root / "instances" / persona / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    if args.vendor:
        vendors = [v.strip() for v in args.vendor.split(",") if v.strip()]
        unknown = [v for v in vendors if v not in VENDORS]
        if unknown:
            print(f"unknown vendor(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
    else:
        key_env = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
                   "openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
        vendors = [v for v, k in key_env.items() if os.environ.get(k)]
    if not vendors:
        print("no vendors to evaluate (no API keys set)", file=sys.stderr)
        return 2

    cases_path = Path(args.cases) if args.cases else project_root / "evals" / "cases.yaml"
    cases = load_cases(cases_path)
    print(f"running {len(cases)} cases against: {', '.join(vendors)}\n")
    results = asyncio.run(run_evals(vendors, cases))

    failed = [r for r in results if not r.passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0
