import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(PROC_DIR / "X_train.csv")
    X_test = pd.read_csv(PROC_DIR / "X_test.csv")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pd.DataFrame(X_train_scaled, columns=X_train.columns) \
        .to_csv(PROC_DIR / "X_train_scaled.csv", index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns) \
        .to_csv(PROC_DIR / "X_test_scaled.csv", index=False)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    logger.info("Scaler ajusté et sauvegardé : %s", MODELS_DIR / "scaler.pkl")


if __name__ == "__main__":
    main()