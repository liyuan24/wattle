"""PyInstaller entrypoint for the standalone Wattle binary."""

from wattle.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
