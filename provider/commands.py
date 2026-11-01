"""Legacy standalone Kiro auth/usage helper.

Hermes now exposes the provider through ``hermes auth ... kiro`` and ``/usage``.
This script remains only as a recovery path for installations upgrading from
pre-native releases:

    python ~/.hermes/plugins/kiro-provider/commands.py {login|status|usage|logout}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:  # direct `python commands.py` run
    sys.path.insert(0, str(_HERE))

def _status_line() -> str:
    from credentials import get_credentials
    creds = get_credentials()
    kind = f"{creds.social_provider or 'social'} login" if creds.is_social else "IdC"
    return (f"Kiro logged in; {kind} region={creds.region}; runtime region={creds.api_region}; "
            f"expires_in={int(creds.expires_at - time.time())}s")


# --- legacy standalone argparse surface --------------------------------------------------

def setup_parser(parser) -> None:
    sub = parser.add_subparsers(dest="kiro_command", required=True)
    login_parser = sub.add_parser("login", help="Sign in to Kiro via IAM Identity Center")
    login_parser.add_argument("--start-url", help="IAM Identity Center URL; omit for interactive setup")
    login_parser.add_argument("--region", help="IAM Identity Center region; omit for interactive setup")
    sub.add_parser("status", help="Show Kiro login status")
    sub.add_parser("usage", help="Show Kiro usage allowances")
    logout_parser = sub.add_parser("logout", help="Forget Hermes' native Kiro login")
    logout_parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation")


def handle(args) -> None:
    command = getattr(args, "kiro_command", "status")
    if command == "login":
        from credentials import login, prompt_login_inputs
        login(*prompt_login_inputs(args.start_url, args.region))
        print("Kiro login saved. New CLI chats can use it immediately; restart only a running Hermes gateway.")
        return
    if command == "usage":
        from client import format_usage, get_usage_limits
        print(format_usage(get_usage_limits()))
        return
    if command == "logout":
        from credentials import logout
        confirmed = getattr(args, "yes", False) or (
            sys.stdin.isatty()
            and input("Forget Hermes' Kiro credentials? [y/N] ").strip().lower() in {"y", "yes"})
        if not confirmed:
            print("Kiro logout cancelled.")
            return
        print("Kiro credentials removed." if logout() else "Kiro was already logged out.")
        return
    print(_status_line())


# --- rescue entrypoint: no Hermes plugin API imports, works when registration fails -------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="commands.py", description="Kiro login/status/usage/logout without the Hermes CLI")
    setup_parser(parser)
    handle(parser.parse_args())


if __name__ == "__main__":
    main()
