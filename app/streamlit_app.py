"""Streamlit entry point:  streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

# lets `streamlit run` work from a plain checkout without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stockanalyzer.ui.app import main  # noqa: E402

main()
