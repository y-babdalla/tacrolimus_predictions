"""This module contains the function to choose the data file and encode the date columns."""

import os

import pandas as pd


def choose_data(file_path: str, date_columns: list[str]) -> pd.DataFrame:
    """This function takes the dataframe and label encodes all date columns.

    Args:
        file_path (str): The path to the data file.
        date_columns (list): List of date columns to encode.

    Returns:
        pd.DataFrame: The dataframe with encoded date columns
    """
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == ".csv":
        df = pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Data file extension should be csv, xls or xlsx")

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
