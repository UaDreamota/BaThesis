import os
import re
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_PARLAM_DATA_DIR = DEFAULT_DATA_DIR / "parlam"

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


def _env_path(*names: str) -> Path | None:
    for name in names:
        raw_path = os.getenv(name)
        if raw_path:
            return _normalize_env_path(raw_path.strip().strip('"').strip("'"))
    return None


def get_data_dir() -> Path:
    return _env_path(
        "DATA_FOLDER",
        "data_folder",
        "DATA_PATH",
        "PROJECT_DATA_PATH",
        "BA_THESIS_DATA_PATH",
    ) or DEFAULT_DATA_DIR


def get_data_path(*parts: str) -> Path:
    return get_data_dir().joinpath(*parts)


def get_parlam_data_dir() -> Path:
    candidate = _env_path("PARLAM_DATA_PATH")
    if candidate is None:
        data_dir = get_data_dir()
        candidate = data_dir if data_dir.name.lower() == "parlam" else data_dir / "parlam"
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
