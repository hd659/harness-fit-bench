"""Shared coding task: a tiny shop package with two planted bugs.

Each harness gets a fresh copy, the same prompt, and is graded by a hidden checker that
it never sees (so editing the visible tests does not help).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

TASK_PROMPT = (
    "This repo is a tiny shop package. `python -m pytest -q` currently fails. "
    "Find and fix the bugs in the source under shop/ so all tests pass. "
    "Do not modify the tests. Keep the change minimal. Reply DONE when finished."
)

FILES = {
    "shop/__init__.py": "",
    "shop/pricing.py": '''
        def apply_discount(price: float, percent: float) -> float:
            """Return price after a percentage discount, rounded to cents."""
            if not 0 <= percent <= 100:
                raise ValueError("percent must be within 0..100")
            return round(price - percent, 2)


        def add_tax(price: float, rate: float = 0.08) -> float:
            return round(price * (1 + rate), 2)
    ''',
    "shop/cart.py": '''
        from .pricing import add_tax, apply_discount


        class Cart:
            def __init__(self):
                self.items = []

            def add(self, name: str, price: float, qty: int = 1) -> None:
                self.items.append((name, price, qty))

            def subtotal(self) -> float:
                return round(sum(price for _, price, qty in self.items), 2)

            def total(self, discount: float = 0, tax_rate: float = 0.08) -> float:
                return add_tax(apply_discount(self.subtotal(), discount), tax_rate)
    ''',
    "tests/test_shop.py": '''
        from shop.cart import Cart
        from shop.pricing import apply_discount


        def test_discount_percent():
            assert apply_discount(200.0, 10) == 180.0


        def test_cart_qty():
            c = Cart()
            c.add("pen", 2.5, qty=4)
            c.add("book", 10.0)
            assert c.subtotal() == 20.0


        def test_total():
            c = Cart()
            c.add("mug", 50.0, qty=2)
            assert c.total(discount=50, tax_rate=0.1) == 55.0
    ''',
}

HIDDEN_CHECK = '''
import sys
sys.path.insert(0, ".")
from shop.pricing import apply_discount, add_tax
from shop.cart import Cart
checks = []
checks.append(apply_discount(80.0, 25) == 60.0)
checks.append(apply_discount(19.99, 0) == 19.99)
try:
    apply_discount(10, 120); checks.append(False)
except ValueError:
    checks.append(True)
c = Cart(); c.add("a", 1.25, qty=3); c.add("b", 4.0, qty=2)
checks.append(c.subtotal() == 11.75)
checks.append(c.total(discount=20, tax_rate=0.0) == 9.4)
checks.append(add_tax(100.0) == 108.0)
print(sum(checks), len(checks))
'''


def make_workspace(root: str = "") -> str:
    """Create a fresh fixture repo; returns its path."""
    ws = tempfile.mkdtemp(prefix="hfb-task-", dir=root or None)
    for rel, src in FILES.items():
        fp = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(src).lstrip("\n"))
    subprocess.run(["git", "init", "-q"], cwd=ws, check=False, capture_output=True)
    return ws


def tests_untouched(ws: str) -> bool:
    fp = os.path.join(ws, "tests/test_shop.py")
    if not os.path.isfile(fp):
        return False
    with open(fp, "r", encoding="utf-8") as f:
        return f.read() == textwrap.dedent(FILES["tests/test_shop.py"]).lstrip("\n")


def grade(ws: str) -> dict:
    """Run the hidden checker inside ws. Returns {passed, total, solved, tests_untouched}."""
    try:
        out = subprocess.run([sys.executable, "-c", HIDDEN_CHECK], cwd=ws, capture_output=True,
                             text=True, timeout=30)
        passed, total = (int(x) for x in out.stdout.split()[-2:])
    except Exception:  # import error, syntax error, timeout
        passed, total = 0, 6
    untouched = tests_untouched(ws)
    return {"passed": passed, "total": total, "tests_untouched": untouched,
            "solved": passed == total and untouched}


def cleanup(ws: str) -> None:
    shutil.rmtree(ws, ignore_errors=True)
