import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/raw.csv")
OUT_DIR = Path("data/processed")
TARGET = "silica_concentrate"
TEST_SIZE = 0.2
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    logger.info("Dataset chargé : %d lignes, %d colonnes", *df.shape)

    # Sécurité : certaines versions de raw.csv comportent une colonne date en tête.
    first_col = df.columns[0]
    if df[first_col].dtype == "object" or first_col.lower() in {"date", "timestamp", "unnamed: 0"}:
        logger.info("Suppression de la colonne non-numérique : %s", first_col)
        df = df.drop(columns=[first_col])

    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET}' absente du dataset.")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_test.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_train.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUT_DIR / "y_test.csv", index=False)

    logger.info("Split terminé : %d train / %d test", len(X_train), len(X_test))


if __name__ == "__main__":
    main()