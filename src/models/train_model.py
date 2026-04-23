import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    X_train = pd.read_csv(PROC_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROC_DIR / "y_train.csv").values.ravel()

    with open(MODELS_DIR / "best_params.pkl", "rb") as f:
        best_params = pickle.load(f)
    logger.info("Entraînement avec les paramètres : %s", best_params)

    model = GradientBoostingRegressor(random_state=RANDOM_STATE, **best_params)
    model.fit(X_train, y_train)

    with open(MODELS_DIR / "trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("Modèle sauvegardé : %s", MODELS_DIR / "trained_model.pkl")


if __name__ == "__main__":
    main()