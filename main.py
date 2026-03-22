from scripts.sampling import sample
from pipeline import Pipeline

MODELS = ["mlknn", "mltsvm"]
p = Pipeline()


def main():
    # for model in MODELS:
    p.run("mlknn")
    p.run("mltsvm")
    # sample()


if __name__ == "__main__":
    main()
