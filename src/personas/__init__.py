"""personas package — persona identity + DI composition.

A persona is a directory under instances/ holding:
  persona.yaml         — name, system_prompt, enabled_connectors, optional model override
  platform.yaml        — platform type + config (e.g. type: telegram)
  .env                 — secrets (TELEGRAM_TOKEN, API keys, DATABASE_URL, …)
  connectors.yaml      — per-profile config for the enabled connectors
  data/                — sessions.json (per-chat session ids)
  credentials/         — per-profile OAuth files & secrets

Persona is the data; PersonaRuntime is the DI factory that builds the
runtime dependency graph for one persona.

Public API:
    from personas import Persona, PersonaRuntime
    persona = Persona.load("personal_assistant", project_root)
    PersonaRuntime(persona).create_conversation().run()
"""
from __future__ import annotations

from .container import PersonaRuntime
from .persona import Persona

__all__ = ["Persona", "PersonaRuntime"]
