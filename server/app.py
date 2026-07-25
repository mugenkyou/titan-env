from __future__ import annotations

from TITAN_env.server.app import create_env, main as _pkg_main


def main() -> int:
    return _pkg_main()


if __name__ == "__main__":
    raise SystemExit(main())
