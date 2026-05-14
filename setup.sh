#!/usr/bin/env bash
set -e

echo "=== Kalshi Bot Setup ==="

# 1. Create venv
python3 -m venv .venv
source .venv/bin/activate

# 2. Install deps
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 3. Create .env if missing
if [ ! -f .env ]; then
  cp .env.example .env
  echo ""
  echo "Created .env — fill in your credentials before running the bot."
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Edit .env with your Kalshi credentials"
echo "  2. source .venv/bin/activate"
echo "  3. python main.py --scan    # test: see opportunities without trading"
echo "  4. python main.py           # run in demo mode"
echo "  5. python main.py --live    # run with real money (edit DEMO_MODE=false in .env first)"
