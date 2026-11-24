"""scheduled_payments — Ahon's payment queue as data for the printed paper."""
import json
from datetime import datetime, timedelta

from adapters.tools.budget import _MANILA, _schedule_tools


class FakeClient:
    def __init__(self, rows):
        self.rows = rows

    async def list_pending_payments(self):
        return {"due": self.rows[:1], "upcoming": self.rows[1:]}


def _d(n):
    return (datetime.now(_MANILA).date() + timedelta(days=n)).isoformat()


async def test_window_direction_and_overdue():
    rows = [
        {"id": 1, "due_date": _d(-2), "amount": "1799.00", "currency": "PHP",
         "description": "Globe Postpaid", "type": "debit", "account_name": "BPI"},
        {"id": 2, "due_date": _d(3), "amount": "50000", "currency": "PHP",
         "description": "Salary", "type": "credit", "account_name": "BPI"},
        {"id": 3, "due_date": _d(40), "amount": "10", "currency": "PHP",
         "description": "Far away", "type": "debit"},
    ]
    spec = _schedule_tools(FakeClient(rows))[0]
    assert spec.name == "scheduled_payments"
    out = json.loads((await spec.handler({"days": 14}, None)).text)
    assert [(r["title"], r["direction"], r["overdue"]) for r in out] == [
        ("Globe Postpaid", "out", True),
        ("Salary", "in", False),
    ]
