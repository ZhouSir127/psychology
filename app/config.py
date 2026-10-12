from dataclasses import dataclass
from pathlib import Path
import os
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Settings:
    root: Path
    data_dir: Path
    options: dict

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.root / path

def load_settings() -> Settings:
    config_path = Path(os.environ.get("MINDTRACK_CONFIG", str(ROOT / "config.toml")) )
    
    with config_path.open("rb") as file:
        options = tomllib.load(file)
    
    data_dir = Path(os.environ.get("MINDTRACK_DATA_DIR", options["app"]["data_dir"]) )
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    
    risk = options["risk"]
    bounds, labels = risk["upper_bounds"], risk["labels"]

    if (not bounds or len(bounds) != len(labels) or sorted(set(bounds)) != bounds
            or bounds[0] < 0 or bounds[-1] != 100):
        raise ValueError("risk.upper_bounds 必须递增、末项为100，并与 labels 等长")

    return Settings(ROOT, data_dir, options)