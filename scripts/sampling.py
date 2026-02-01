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

scaler_metrics: list[dict] = []


def sample() -> None:
    X = df.drop(columns=["Activity"], axis=1)
    y = df["Activity"]
    for idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        train_frms, scaler_metrics_i, scaler = generate_frms(df.iloc[train_idx])
        test_frms, _, _ = generate_frms(df.iloc[test_idx], scaler)

        for feature in train_frms.keys():
            train_df = train_frms[feature]
            test_df = test_frms[feature]
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

        scaler_metrics.append(scaler_metrics_i)

    with open(ROOT + "pcp_normalization.json", "w") as f:
        ujson.dump(scaler_metrics, f, indent=4)
    with open(ROOT + "sampling_metrics.json", "w") as f:
        ujson.dump(metrics, f, indent=4)

    print("Sampling completed")
    print(f"Splits generated: {SPLITS}")
    print(f"Test size: {TEST_SIZE * 100}%")
    print(f"Random state: {RANDOM_STATE}")
