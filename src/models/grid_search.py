import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RANDOM_STATE = 42

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(PROC_DIR / "X_train_scaled.csv")
    y_train = pd.read_csv(PROC_DIR / "y_train.csv").values.ravel()

    base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    logger.info("Meilleurs paramètres : %s", grid.best_params_)
    logger.info("Meilleur R² (cv)     : %.4f", grid.best_score_)

    with open(MODELS_DIR / "best_params.pkl", "wb") as f:
        pickle.dump(grid.best_params_, f)


if __name__ == "__main__":
    main()