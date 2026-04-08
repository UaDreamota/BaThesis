import os
import re
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARLAM_DATA_DIR = REPO_ROOT / "data" / "parlam"

load_dotenv(REPO_ROOT / ".env")


def _looks_like_parlam_dir(path: Path) -> bool:
    return any(path.glob("ParlaMint-*_extracted.csv")) or any(path.glob("ParlaMint-*"))


def _normalize_env_path(raw_path: str) -> Path:
    windows_drive_match = re.match(r"^([A-Za-z]):[\\/](.*)$", raw_path)
    if os.name != "nt" and windows_drive_match:
        drive = windows_drive_match.group(1).lower()
        tail = windows_drive_match.group(2).replace("\\", "/")
        return Path(f"/mnt/{drive}/{tail}")

    return Path(raw_path).expanduser()


def get_parlam_data_dir() -> Path:
    raw_path = os.getenv("PARLAM_DATA_PATH")
    if not raw_path:
        return DEFAULT_PARLAM_DATA_DIR

    candidate = _normalize_env_path(raw_path)
    if candidate.name.lower() == "parlam":
        return candidate

    if candidate.exists() and _looks_like_parlam_dir(candidate):
        return candidate

    parlam_child = candidate / "parlam"
    if parlam_child.exists() or (candidate.exists() and not _looks_like_parlam_dir(candidate)):
        return parlam_child

    return candidate


def get_parlam_csv_path(country_code: str) -> Path:
    return get_parlam_data_dir() / f"ParlaMint-{country_code}_extracted.csv"
