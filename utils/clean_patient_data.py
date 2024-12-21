from datetime import datetime
import pandas as pd


def clean_patient_data(df, date_columns):
    """
    Function to clean syntheic data
    :param df: the dataframe to be cleaned
    :param data_type: type of data, could be followup or patient
    :return:
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Define the cutoff date as March 2023
    cutoff_date = datetime.strptime('2023-03-31', '%Y-%m-%d')

    if 'Fecha de seguimiento' in df.columns:
        df['Fecha de seguimiento'] = pd.to_datetime(df['Fecha de seguimiento'], errors='coerce')

        # Remove rows where dates in "Fecha de seguimiento" are after March 2023
        df = df[df['Fecha de seguimiento'] <= cutoff_date]
    
    if 'fecha' in df.columns:
        # Remove rows where dates in "fecha" are after March 2023
        df = df[df['fecha'] <= cutoff_date]

    if {'exitus', 'fechaExitus'}.issubset(df.columns):
        # Remove rows where 'exitus' is 'Si' and 'fechaExitus' is either after the cutoff date or NaN
        df = df[~((df['exitus'] == 'Si') & ((df['fechaExitus'] > cutoff_date) | pd.isna(df['fechaExitus'])))]

        # Remove rows where 'exitus' is 'No' and 'fechaExitus' is not NaN
        df = df[~((df['exitus'] == 'No') & pd.notna(df['fechaExitus']))]

    if {'Estado', 'Fecha exitus', 'Causa exitus', 'Estado_Exitus'}.issubset(df.columns):
        # Remove rows where 'Estado' is 'Exitus' and 'Fecha exitus' is either after the cutoff date or NaN
        df = df[~((df['Estado'] == 'Exitus') & ((df['Fecha exitus'] > cutoff_date) | pd.isna(df['Fecha exitus'])))]

        # Remove rows where 'Estado' is not 'Exitus' and 'Fecha exitus' is not NaN
        df = df[~((df['Estado'] != 'Exitus') & pd.notna(df['Fecha exitus']))]

        # Remove rows where 'Estado' is not 'Exitus' and 'Causa exitus' is not NaN
        df = df[~((df['Estado'] != 'Exitus') & pd.notna(df['Causa exitus']))]

        # Remove rows where 'Estado' is not 'Exitus' and 'Estado_Exitus' is 1
        df = df[~((df['Estado'] != 'Exitus') & (df['Estado_Exitus'] == 1))]

    if 'Fecha de intervención' in df.columns:
        # Remove rows where dates in 'Fecha de intervención' are after March 2023
        df = df[df['Fecha de intervención'] <= cutoff_date]

    return df

