import logging
import os
from pathlib import Path
from urllib.request import urlretrieve

RAW_URL = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "raw.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        logger.info("Fichier déjà présent : %s (on ne re-télécharge pas).", RAW_FILE)
        return
    logger.info("Téléchargement depuis %s ...", RAW_URL)
    urlretrieve(RAW_URL, RAW_FILE)
    logger.info("Fichier sauvegardé : %s (%d octets)", RAW_FILE, os.path.getsize(RAW_FILE))


if __name__ == "__main__":
    main()