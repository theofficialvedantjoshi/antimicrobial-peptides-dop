import pandas as pd
import ujson
from sklearn.model_selection import StratifiedShuffleSplit

from frm import generate_frms

ROOT = "data/"
SPLITS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

sss = StratifiedShuffleSplit(
    n_splits=SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

metrics = {"splits": SPLITS, "test_size": TEST_SIZE, "random_state": RANDOM_STATE}

df = pd.read_csv(ROOT + "raw_data.csv")

frms, scaler_metrics = generate_frms(df)


def sample() -> None:
    for feature in frms.keys():
        X = frms[feature].drop(columns=["Activity"], axis=1)
        y = frms[feature]["Activity"]
        for idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
            train_df = frms[feature].iloc[train_idx]
            test_df = frms[feature].iloc[test_idx]

            train_df.to_csv(
                ROOT + f"train/{feature}/train_{feature}_{idx+1}.csv", index=False
            )
            test_df.to_csv(
                ROOT + f"test/{feature}/test_{feature}_{idx+1}.csv", index=False
            )

            metrics["train_distribution"] = (
                train_df["Activity"].value_counts(normalize=True).to_dict()
            )
            metrics["test_distribution"] = (
                test_df["Activity"].value_counts(normalize=True).to_dict()
            )

    with open(ROOT + "pcp_normalization.json", "w") as f:
        ujson.dump(scaler_metrics, f, indent=4)
    with open(ROOT + "sampling_metrics.json", "w") as f:
        ujson.dump(metrics, f, indent=4)

    print("Sampling completed")
    print(f"Splits generated: {SPLITS}")
    print(f"Test size: {TEST_SIZE * 100}%")
    print(f"Random state: {RANDOM_STATE}")
