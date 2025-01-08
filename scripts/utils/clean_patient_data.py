"""Module to clean synthetic data."""

from datetime import datetime

import pandas as pd


def clean_patient_data(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """Function to clean synthetic data.

    Note that the data set in this case is from Spain, hence we need to clean the data accordingly.

    Args:
        df (pd.DataFrame): The DataFrame containing the synthetic data.
        date_columns (list[str]): The list of columns containing dates.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    cutoff_date = datetime.strptime("2023-03-31", "%Y-%m-%d")

    if "Fecha de seguimiento" in df.columns:
        df["Fecha de seguimiento"] = pd.to_datetime(df["Fecha de seguimiento"], errors="coerce")

        df = df[df["Fecha de seguimiento"] <= cutoff_date]

    if "fecha" in df.columns:
        df = df[df["fecha"] <= cutoff_date]

    if {"exitus", "fechaExitus"}.issubset(df.columns):
        df = df[
            ~(
                (df["exitus"] == "Si")
                & ((df["fechaExitus"] > cutoff_date) | pd.isna(df["fechaExitus"]))
            )
        ]
        df = df[~((df["exitus"] == "No") & pd.notna(df["fechaExitus"]))]

    if {"Estado", "Fecha exitus", "Causa exitus", "Estado_Exitus"}.issubset(df.columns):
        df = df[
            ~(
                (df["Estado"] == "Exitus")
                & ((df["Fecha exitus"] > cutoff_date) | pd.isna(df["Fecha exitus"]))
            )
        ]
        df = df[~((df["Estado"] != "Exitus") & pd.notna(df["Fecha exitus"]))]
        df = df[~((df["Estado"] != "Exitus") & pd.notna(df["Causa exitus"]))]
        df = df[~((df["Estado"] != "Exitus") & (df["Estado_Exitus"] == 1))]

    if "Fecha de intervención" in df.columns:
        df = df[df["Fecha de intervención"] <= cutoff_date]

    return df
