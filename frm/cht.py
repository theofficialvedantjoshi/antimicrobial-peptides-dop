GROUPS = {
    "Q": 1,
    "P": 1,
    "N": 1,
    "A": 1,
    "T": 1,
    "S": 1,
    "V": 1,
    "G": 1,
    "Y": 2,
    "W": 2,
    "F": 2,
    "E": 3,
    "D": 3,
    "K": 4,
    "H": 4,
    "R": 4,
    "M": 5,
    "C": 5,
    "I": 5,
    "L": 5,
}


def str_key(sequence: str) -> tuple[str, int, str]:
    return (
        "".join(str(GROUPS.get(aa, "0")) for aa in sequence),
        len(sequence) - 2,
        sequence,
    )


def calculate_cht(sequence: str) -> dict[str, float]:
    if len(sequence) < 3:
        return {
            f"{i}{j}{k}": 0.0
            for i in range(1, 6)
            for j in range(1, 6)
            for k in range(1, 6)
        }

    peptide, counter, _ = str_key(sequence)
    return {
        f"{i}{j}{k}": sum(
            1 for m in range(len(peptide)) if peptide.startswith(f"{i}{j}{k}", m)
        )
        / counter
        for i in range(1, 6)
        for j in range(1, 6)
        for k in range(1, 6)
    }
