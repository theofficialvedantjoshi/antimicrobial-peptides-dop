import pandas as pd
import ujson
from sklearn.model_selection import StratifiedShuffleSplit

from frm import generate_frms

ROOT = "data/"
SPLITS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv(ROOT + "raw_data.csv")

X = df.drop(columns=["Activity"], axis=1)
y = df["Activity"]

sss = StratifiedShuffleSplit(
    n_splits=SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

metrics = {"splits": SPLITS, "test_size": TEST_SIZE, "random_state": RANDOM_STATE}


def sample() -> None:
    for idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        train_frms, train_scaler_metrics = generate_frms(train_df)
        test_frms, test_scaler_metrics = generate_frms(test_df)

        for key in train_frms.keys():
            train_frms[key].to_csv(
                ROOT + f"train/{key}/train_{key}_{idx+1}.csv", index=False
            )
            test_frms[key].to_csv(
                ROOT + f"test/{key}/test_{key}_{idx+1}.csv", index=False
            )

        with open(
            ROOT + f"train/pcp/normalization/scaler_metrics_{idx+1}.json", "w"
        ) as f:
            ujson.dump(train_scaler_metrics, f, indent=4)

        with open(
            ROOT + f"test/pcp/normalization/scaler_metrics_{idx+1}.json", "w"
        ) as f:
            ujson.dump(test_scaler_metrics, f, indent=4)

        metrics["train_distribution"] = (
            train_df["Activity"].value_counts(normalize=True).to_dict()
        )
        metrics["test_distribution"] = (
            test_df["Activity"].value_counts(normalize=True).to_dict()
        )

    with open(ROOT + "sampling_metrics.json", "w") as f:
        ujson.dump(metrics, f, indent=4)

    print("Sampling completed")
    print(f"Splits generated: {SPLITS}")
    print(f"Test size: {TEST_SIZE * 100}%")
    print(f"Random state: {RANDOM_STATE}")
