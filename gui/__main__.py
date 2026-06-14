"""
gui/__main__.py

Run from your project root with either:
    python -m gui
    python gui/__main__.py
"""

import sys
import os

# Ensure the project root is on sys.path so all sibling packages
# (models, pipeline, translation, etc.) are importable regardless
# of how this file is invoked.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.model_manager import ModelManager
from gui.main_window import MainWindow


def main():
    print("\nLoading models, please wait...\n")
    manager = ModelManager()
    manager.load_models()
    app = MainWindow(manager)
    app.run()


if __name__ == "__main__":
    main()