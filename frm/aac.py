from collections import Counter


def calculate_aac(sequence: str) -> dict[str, float]:
    amino_acids = "ARNDCEQGHILKMFPSTWYV"
    counts = Counter(sequence)
    return {aa: counts[aa] / len(sequence) for aa in amino_acids}
