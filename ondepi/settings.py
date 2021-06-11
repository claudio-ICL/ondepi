import os
from pathlib import Path

home_path = Path(__file__).resolve().parent
resources_path = home_path / "resources"
intensity_path = resources_path / "intensity"
data_path = home_path.parent / "data"
