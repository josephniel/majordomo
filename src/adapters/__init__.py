"""Adapters — concrete implementations of the ports.

Each subpackage adapts ONE external reality to one port:

    kernel/    chat platforms (telegram, ...)   -> ChatPlatform
    model/   LLM vendors + failover           -> Agent
    tools/   external services (gmail, ...)   -> ToolProvider
    trigger/ time and event sources           -> runtime services
    store/   Postgres persistence             -> memory/doc stores
    adapters/comms/   the shared inter-bot comms bus

Adapters never import each other; anything two of them need to
share belongs in `ports` as a contract. `store` and `comms` sit a
tier lower than the rest — they are infrastructure the other
adapters are allowed to build on.
"""
