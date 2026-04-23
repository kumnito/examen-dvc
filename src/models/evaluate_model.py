import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = pd.read_csv(PROC_DIR / "X_test_scaled.csv")
    y_test = pd.read_csv(PROC_DIR / "y_test.csv").values.ravel()

    with open(MODELS_DIR / "trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)

    pd.DataFrame({"y_true": y_test, "y_pred": predictions}) \
        .to_csv(PROC_DIR / "predictions.csv", index=False)

    mse = mean_squared_error(y_test, predictions)
    scores = {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }
    with open(METRICS_DIR / "scores.json", "w") as f:
        json.dump(scores, f, indent=4)

    logger.info("Scores : %s", scores)


if __name__ == "__main__":
    main()