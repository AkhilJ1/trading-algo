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
Just run it and answer the prompts (the secret is hidden as you type):

    python schwab_auth.py

It asks for your App Key, App Secret, and callback URL, then prints a Schwab
login URL. Log in, approve, and you'll be redirected to your callback URL (it may
look like a browser error page — that's fine, nothing is listening there). Copy
the FULL redirected URL from the address bar and paste it back here. The token
file is then written/refreshed and you're set for another ~7 days.

You can also pre-set any of these as environment variables to skip the prompts
(handy for re-runs): SCHWAB_API_KEY, SCHWAB_APP_SECRET, SCHWAB_CALLBACK_URL,
SCHWAB_TOKEN_PATH (defaults to schwab_token.json).

NOTE ON SAFETY
--------------
This script only ever runs locally, on a machine where you are logged into your
own Schwab account in a browser. The hidden prompt keeps your secret out of your
shell history. It writes the token to SCHWAB_TOKEN_PATH, which must stay out of
git (see .gitignore). Never commit the token or your secret.
"""

import os
import sys
import getpass


def _prompt(env_name: str, label: str, *, secret: bool = False, default: str = None) -> str:
    """Use the env var if set; otherwise ask interactively."""
    val = os.environ.get(env_name)
    if val:
        return val.strip()
    suffix = f" [{default}]" if default else ""
    if secret:
        entered = getpass.getpass(f"{label}{suffix}: ").strip()
    else:
        entered = input(f"{label}{suffix}: ").strip()
    return entered or (default or "")


def main() -> int:
    print("Schwab authentication — answer the prompts (secret input is hidden).\n")
    api_key = _prompt('SCHWAB_API_KEY', 'App Key (Client ID)')
    app_secret = _prompt('SCHWAB_APP_SECRET', 'App Secret', secret=True)
    callback_url = _prompt('SCHWAB_CALLBACK_URL', 'Callback URL', default='https://127.0.0.1')
    token_path = os.environ.get('SCHWAB_TOKEN_PATH', 'schwab_token.json')

    if not api_key or not app_secret:
        print("ERROR: App Key and App Secret are both required.")
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
