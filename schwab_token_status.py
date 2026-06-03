"""
Schwab token-expiry notifier (Requirement 2: autonomous, with a heads-up before
the one weekly human step).

Schwab's refresh token lives ~7 days and does NOT rotate when used — after that a
human must run `python schwab_auth.py` (browser login). This script reads the
token file written by schwab-py, computes how long the refresh token has left,
and signals when a re-auth is due soon. The reauth-check GitHub workflow runs it
daily and opens a GitHub issue (which emails you) when it returns "needed".

Exit codes (so a workflow can branch on them):
    0  = fine, OR Schwab not configured yet (no token) — nothing to do.
    10 = re-auth needed soon (within WARN_WITHIN_HOURS).

The schwab-py token file wraps the OAuth payload with a top-level
`creation_timestamp` (epoch seconds) marking when the refresh token was issued;
that's the clock we measure against. If it's missing we fall back to file mtime.
"""

import os
import sys
import json
from datetime import datetime, timedelta, timezone

REFRESH_TOKEN_TTL_DAYS = 7
WARN_WITHIN_HOURS = 36  # notify when ~1.5 days or less remain


def main() -> int:
    token_path = os.environ.get("SCHWAB_TOKEN_PATH", "schwab_token.json")

    if not os.path.exists(token_path):
        print("NO_TOKEN: Schwab not configured (no token file) — skipping.")
        return 0

    try:
        with open(token_path, "r") as f:
            tok = json.load(f)
    except Exception as e:
        # A corrupt token file effectively means re-auth is required.
        print(f"REAUTH_NEEDED: token file unreadable ({e}).")
        return 10

    created = tok.get("creation_timestamp")
    if created is None:
        created = os.path.getmtime(token_path)

    created_dt = datetime.fromtimestamp(float(created), tz=timezone.utc)
    age = datetime.now(timezone.utc) - created_dt
    remaining = timedelta(days=REFRESH_TOKEN_TTL_DAYS) - age
    hours_left = remaining.total_seconds() / 3600.0

    print(
        f"Schwab refresh token issued {created_dt.isoformat()} | "
        f"age {age.days}d {age.seconds // 3600}h | ~{hours_left:.1f}h remaining "
        f"of {REFRESH_TOKEN_TTL_DAYS}d."
    )

    if hours_left <= WARN_WITHIN_HOURS:
        print("REAUTH_NEEDED")
        return 10

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
