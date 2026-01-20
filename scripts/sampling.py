import pandas as pd
import ujson
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = "../data/"
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

for idx, (train_idx, test_idx) in enumerate(sss.split(X, y)):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    metrics["train_distribution"] = (
        train_df["Activity"].value_counts(normalize=True).to_dict()
    )
    metrics["test_distribution"] = (
        test_df["Activity"].value_counts(normalize=True).to_dict()
    )

    train_df.to_csv(ROOT + f"train/train_{idx+1}.csv", index=False)
    test_df.to_csv(ROOT + f"test/test_{idx+1}.csv", index=False)

with open(ROOT + "sampling_metrics.json", "w") as f:
    ujson.dump(metrics, f, indent=4)

print("Sampling completed")
print(f"Splits generated: {SPLITS}")
print(f"Test size: {TEST_SIZE * 100}%")
print(f"Random state: {RANDOM_STATE}")
