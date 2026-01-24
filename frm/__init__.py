import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from frm.aac import calculate_aac
from frm.cht import calculate_cht
from frm.pcp import PCP_FEATURES, calculate_pcp


def generate_frms(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict]:
    aac_df = pd.DataFrame(df["Sequence"].apply(calculate_aac).to_list())
    cht_df = pd.DataFrame(df["Sequence"].apply(calculate_cht).to_list())

    pcp_df = pd.DataFrame(df["Sequence"].apply(calculate_pcp).to_list())
    scaler = MinMaxScaler()
    pcp_df = pd.DataFrame(scaler.fit_transform(pcp_df), columns=pcp_df.columns)

    combination_df = pd.concat([df, aac_df, cht_df, pcp_df], axis=1)
    combination_df["Activity"] = combination_df.pop("Activity")

    aac_df = pd.concat([df, aac_df], axis=1)
    aac_df["Activity"] = aac_df.pop("Activity")
    cht_df = pd.concat([df, cht_df], axis=1)
    cht_df["Activity"] = cht_df.pop("Activity")
    pcp_df = pd.concat([df, pcp_df], axis=1)
    pcp_df["Activity"] = pcp_df.pop("Activity")

    scaler_metrics = {
        "min": dict(zip(PCP_FEATURES, scaler.data_min_)),
        "max": dict(zip(PCP_FEATURES, scaler.data_max_)),
    }
    return {
        "aac": aac_df,
        "cht": cht_df,
        "pcp": pcp_df,
        "combination": combination_df,
    }, scaler_metrics
