import sys
from pathlib import Path

import pytest

try:  # pragma: no cover - optional dependency in test env
    from PySide6.QtWidgets import QApplication
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    QApplication = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def qapp():
    if QApplication is None:
        pytest.skip("PySide6 not installed")
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
