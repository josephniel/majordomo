"""Eval harnesses.

Two, measuring the two ways this assistant fails silently:

    runner.py — vendor TOOL-CALLING replay. Does the model call the right
                tool for a given message? Regresses on every vendor/model
                swap.
    recall.py — memory RECALL quality. Does the right fact surface for a
                given query? Regresses on every retrieval change.
"""
