"""Day 85 — the only door to the test rows.

The first call to open_test() writes a lock file holding a fingerprint of
the recipe files. The lock is committed with the results. Every later call
must present the same fingerprint: rerunning the identical recipe is
allowed (it reproduces the same numbers), but a changed recipe is refused,
because tuning after seeing test would turn it into a second validation set.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


class HoldoutAlreadyOpened(RuntimeError):
    """Raised when test was opened under one recipe and is requested under another."""


def fingerprint(paths) -> str:
    """sha256 over the named files, in the order given, each prefixed by its name."""
    digest = hashlib.sha256()
    for path in paths:
        path = Path(path)
        # the name goes in too, so swapping two files' contents changes the hash
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def read_lock(lock_path) -> dict | None:
    lock_path = Path(lock_path)
    if not lock_path.exists():
        return None
    fields = {}
    for line in lock_path.read_text().splitlines():
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    if "fingerprint" not in fields:
        raise ValueError(f"{lock_path} has no fingerprint line")
    return fields


def open_test(parts: dict, lock_path, recipe_fingerprint: str, now: datetime | None = None):
    """Return the test rows, recording or checking the lock first.

    Returns (test_df, first_time). first_time is True only on the run that
    wrote the lock.
    """
    if "test" not in parts:
        raise KeyError("parts has no test slice")
    lock = read_lock(lock_path)
    if lock is not None:
        if lock["fingerprint"] != recipe_fingerprint:
            raise HoldoutAlreadyOpened(
                f"test was opened on {lock.get('opened', '?')} with recipe {lock['fingerprint'][:12]}…; "
                f"the recipe is now {recipe_fingerprint[:12]}…. Changing it after seeing test isn't allowed."
            )
        return parts["test"], False

    now = now or datetime.now(timezone.utc)
    test = parts["test"]
    Path(lock_path).write_text(
        "# written the first time the test slice was opened; commit it with the results\n"
        f"fingerprint: {recipe_fingerprint}\n"
        f"opened: {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"rows: {len(test)}\n"
    )
    return test, True
