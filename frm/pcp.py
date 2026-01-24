import numpy as np
import ujson
from modlamp.descriptors import GlobalDescriptor

PCP_FEATURES = [
    "length",
    "charge",
    "charge_density",
    "pI",
    "instability_index",
    "aromaticity",
    "aliphatic_index",
    "boman_index",
    "hydrophobic_ratio",
    "hydrophobicity",
    "hydropathy_index",
    "amphiphilicity_index",
    "solvation",
    "steric",
    "volume",
    "side_chain_vol",
    "polarity",
    "no_hyd_bond",
    "norm_freq_alpha",
    "norm_freq_beta",
    "norm_freq_coil",
]


def get_properties() -> list[dict]:
    with open("frm/pcp_properties.json", "r") as f:
        properties = ujson.load(f)
    return properties


def other_properties(sequence: str, map: dict, excl: bool = False) -> float:
    total = sum(map[i] for i in sequence.strip())
    return total if excl else total / len(sequence.strip())


def hydrophobicity(sequence: str) -> float:
    map = {
        "A": -0.17,
        "R": -0.81,
        "N": -0.42,
        "D": -1.23,
        "C": 0.24,
        "Q": -0.58,
        "E": -2.02,
        "G": -0.01,
        "H": -0.96,
        "I": 0.31,
        "L": 0.56,
        "K": -0.99,
        "M": 0.23,
        "F": 1.13,
        "P": -0.45,
        "S": -0.13,
        "T": -0.14,
        "W": 1.85,
        "Y": 0.94,
        "V": -0.07,
    }
    return sum(map[i] for i in sequence.strip()) / len(sequence.strip())


def calculate_pcp(sequence: str) -> dict[str, float]:
    desc = GlobalDescriptor(sequence)
    desc.calculate_all(amide=False)
    gobal_desc = desc.descriptor.tolist()
    gobal_desc = np.delete(gobal_desc, [1]).tolist()  # remove molecular weight
    gobal_desc.append(hydrophobicity(sequence))
    properties = get_properties()
    for i, val in enumerate(properties):
        excl = i in {4, 7}
        gobal_desc.append(other_properties(sequence, val, excl))
    return dict(zip(PCP_FEATURES, gobal_desc))
