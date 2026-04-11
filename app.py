"""
HuggingFace Spaces entrypoint.
This file must be named app.py and live at the project root for HF Spaces.
"""
import subprocess
import sys
from pathlib import Path

# Bootstrap: load seed data if DB is empty
sys.path.insert(0, str(Path(__file__).parent))

from src.db.repository import BenchmarkRepository
from scripts.load_seed_data import load_seed_csv

repo = BenchmarkRepository()
if repo.count() == 0:
    load_seed_csv(repo)
repo.close()

# Launch the app
from src.ui.app import build_app

app = build_app()
app.launch()
