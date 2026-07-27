"""Assemble the full Schwab block for Streamlit secrets and copy it to the clipboard.

Rebuilds every Schwab key at once so a partial paste can't drop SCHWAB_API_KEY or
SCHWAB_APP_SECRET (which is exactly how the live app ended up with a valid token
but a failing client build). Nothing is echoed to the terminal and nothing is
written to disk — the block goes straight to the pasteboard via pbcopy.
"""

import getpass
import json
import subprocess
import sys

TOKEN_PATH = "schwab_token.json"


def main() -> int:
    try:
        with open(TOKEN_PATH) as f:
            token = f.read().strip()
        json.loads(token)  # fail loudly here rather than in the cloud
    except FileNotFoundError:
        print(f"ERROR: {TOKEN_PATH} not found. Run schwab_auth.py first.")
        return 1
    except json.JSONDecodeError as e:
        print(f"ERROR: {TOKEN_PATH} is not valid JSON ({e}). Re-run schwab_auth.py.")
        return 1

    print("Paste your Schwab app credentials (input is hidden).\n")
    api_key = getpass.getpass("App Key (Client ID): ").strip()
    app_secret = getpass.getpass("App Secret: ").strip()

    if not api_key or not app_secret:
        print("ERROR: both App Key and App Secret are required.")
        return 1

    # Triple-quoted TOML literals keep the token byte-for-byte; any other
    # quoting lets TOML mangle characters inside the refresh token, which
    # parses fine but dies at the first refresh with invalid_grant.
    block = (
        f"SCHWAB_API_KEY = '''{api_key}'''\n"
        f"SCHWAB_APP_SECRET = '''{app_secret}'''\n"
        f"SCHWAB_TOKEN_PATH = '''schwab_token.json'''\n"
        f"DATA_PROVIDER = '''schwab'''\n"
        f"SCHWAB_TOKEN = '''{token}'''\n"
    )

    subprocess.run(["pbcopy"], input=block, text=True, check=True)
    print(f"\n✅ Copied {len(block)} chars — 5 keys — to the clipboard.")
    print("   Replace the whole Schwab section of Streamlit secrets, then Reboot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
