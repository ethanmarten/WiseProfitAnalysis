"""Compatibility entry point for the maintained local MT5 bridge.

Run ``python local/run_local.py`` from the repository root, or run
``python bridge/local_mt5_bot.py`` directly. Keeping one implementation avoids
different authentication and signal-acknowledgement behaviour between modes.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bridge.local_mt5_bot import main


if __name__ == "__main__":
    main()
