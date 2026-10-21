#!/usr/bin/env bash
set -e
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_dummy_data.py
pytest -q
printf '\nSetup complete. Start the app with: streamlit run app.py\n'
