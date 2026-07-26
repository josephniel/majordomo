"""Module entry point: `python -m chat --persona <id>`.

Owns argument parsing + persona loading + composition. Importing Persona
and PersonaRuntime here (instead of from kernel/core.py) keeps the cycle
between chat and personas closed: chat.core doesn't depend on personas
at all, so personas.container can import ConversationOrchestrator from chat.core without
ceremony.
"""
import argparse
import logging
from pathlib import Path

from runtime import Persona, PersonaRuntime

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # httpx logs every request URL at INFO — for Telegram polling that URL
    # embeds the bot token, which must never land in logs/.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    project_root = Path(__file__).resolve().parent.parent.parent
    parser = argparse.ArgumentParser(description="Run the given persona on its configured chat platform (LLM-agnostic).")
    parser.add_argument(
        "--persona",
        required=True,
        help="persona id (directory name under instances/, e.g. personal_assistant)",
    )
    args = parser.parse_args()

    persona = Persona.load(args.persona, project_root)
    log.info("loading persona: %s (%s)", persona.id, persona.name)
    PersonaRuntime(persona).create_conversation().run()


if __name__ == "__main__":
    main()
