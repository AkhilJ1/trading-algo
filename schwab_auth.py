"""
Schwab OAuth helper — (re)create the local token file.

WHY THIS EXISTS
---------------
Schwab's API uses two tokens:
  * an access token that auto-refreshes for ~7 days, and
  * a refresh token that does NOT rotate and expires after 7 days.
So roughly once a week a human has to log in through the browser to mint a fresh
token. This script is that one command. Everything else (the dashboard, the
daily recorder) reads the token file silently and never needs a browser.

USAGE
-----
    # one-time setup: put your app credentials in the environment
    export SCHWAB_API_KEY="...."
    export SCHWAB_APP_SECRET="...."
    export SCHWAB_CALLBACK_URL="https://127.0.0.1"   # must match your app config
    export SCHWAB_TOKEN_PATH="schwab_token.json"      # optional, this is default

    python schwab_auth.py

It prints a Schwab login URL. Log in, approve, and you'll be redirected to your
callback URL (it may look like an error page — that's fine). Copy the FULL
redirected URL from the browser address bar and paste it back here. The token
file is then written/refreshed and you're set for another ~7 days.

NOTE ON SAFETY
--------------
This script only ever runs locally, on a machine where you are logged into your
own Schwab account in a browser. It writes the token to SCHWAB_TOKEN_PATH, which
must stay out of git (see .gitignore). Never commit the token or your secret.
"""

import os
import sys


def main() -> int:
    api_key = os.environ.get('SCHWAB_API_KEY')
    app_secret = os.environ.get('SCHWAB_APP_SECRET')
    callback_url = os.environ.get('SCHWAB_CALLBACK_URL', 'https://127.0.0.1')
    token_path = os.environ.get('SCHWAB_TOKEN_PATH', 'schwab_token.json')

    missing = [
        name for name, val in (
            ('SCHWAB_API_KEY', api_key),
            ('SCHWAB_APP_SECRET', app_secret),
        ) if not val
    ]
    if missing:
        print(f"ERROR: missing env var(s): {', '.join(missing)}")
        print("Set them and re-run. See the docstring at the top of this file.")
        return 1

    try:
        # Manual flow: prints a URL, you paste back the redirect. No local
        # HTTPS server / cert juggling required.
        from schwab.auth import client_from_manual_flow
    except ImportError:
        print("ERROR: schwab-py is not installed. Run: pip install schwab-py")
        return 1

    print(f"Authenticating Schwab app — token will be written to: {token_path}")
    print(f"Using callback URL: {callback_url}\n")

    client_from_manual_flow(
        api_key=api_key,
        app_secret=app_secret,
        callback_url=callback_url,
        token_path=token_path,
    )

    if os.path.exists(token_path):
        print(f"\n✅ Success. Token saved to {token_path}.")
        print("   Good for ~7 days. Re-run this script when fetches start "
              "falling back to yfinance.")
        return 0

    print("\n❌ Token file was not created. Check the pasted URL and try again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
