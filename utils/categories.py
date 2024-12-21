import os
import pandas as pd

def choose_data(file_path, date_columns):
    """
    This function takes the dataframe and label encodes all date columns
    :param df: dataframe to be encoded
    :param file_path: path to the data file
    :return: encoded dataframe
    """
    file_extension = os.path.splitext(file_path)[-1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError('Data file extension should be csv, xls or xlsx')

    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df
