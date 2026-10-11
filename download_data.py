"""
Downloads the real-world 5G production dataset used across all three ZTM
ML tasks in this project.

Source (real, publicly available, collected on a live Irish 5G/LTE
network by University College Cork's Mobile & Internet Systems Lab):

    https://github.com/uccmisl/5Gdataset

Reference:
    D. Raca, D. Leahy, C.J. Sreenan and J.J. Quinlan. "Beyond Throughput,
    The Next Generation: A 5G Dataset with Channel and Context Metrics."
    ACM MMSys 2020.

The repo stores its data as a zip file (5G-production-dataset.zip)
containing ~166 CSV traces of real client-side KPIs (RSRP, RSRQ, SNR,
CQI, RSSI, DL/UL bitrate, GPS, ping stats, ...) captured while streaming
Netflix / Amazon Prime / doing file downloads, both stationary and
while driving.

Run standalone:
    python src/download_data.py
"""
import subprocess
import zipfile
from pathlib import Path

REPO_URL = "https://github.com/uccmisl/5Gdataset.git"
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLONE_DIR = RAW_DIR / "5Gdataset_repo"
EXTRACT_DIR = RAW_DIR / "5G-production-dataset"


def download_and_extract(force: bool = False) -> Path:
    """Clones the repo (shallow) and extracts the dataset zip inside it.
    Returns the path to the extracted CSV tree.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists() and any(EXTRACT_DIR.rglob("*.csv")) and not force:
        print(f"[download_data] Dataset already present at {EXTRACT_DIR}, skipping.")
        return EXTRACT_DIR

    if not CLONE_DIR.exists():
        print(f"[download_data] Cloning {REPO_URL} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)],
            check=True,
        )
    else:
        print(f"[download_data] Repo already cloned at {CLONE_DIR}.")

    zip_path = CLONE_DIR / "5G-production-dataset.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Expected dataset zip not found at {zip_path}. "
            "The upstream repo layout may have changed."
        )

    print(f"[download_data] Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)

    n_csv = len(list(EXTRACT_DIR.rglob("*.csv")))
    print(f"[download_data] Done. {n_csv} CSV traces available at {EXTRACT_DIR}")
    return EXTRACT_DIR


if __name__ == "__main__":
    download_and_extract()
