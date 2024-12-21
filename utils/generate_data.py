import os

import pandas as pd
from sdv.lite import SingleTablePreset
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdmetrics.single_column import RangeCoverage, CategoryCoverage, StatisticSimilarity
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    GaussianCopulaSynthesizer,
    CTGANSynthesizer,
    TVAESynthesizer,
    CopulaGANSynthesizer,
)

from utils.clean_patient_data import clean_patient_data

def evaluate_synthetic_data(synthetic_data, real_data, metadata, prefix, out_directory, model):
    # Generate quality metrics - these do not seem to correlate well with how good the data is
    quality_report = evaluate_quality(real_data, synthetic_data, metadata)#train
    os.makedirs(f"{out_directory}/", exist_ok=True)
    quality_report.get_details(property_name='Column Shapes').to_csv(f"{out_directory}/column_shapes_{prefix}_{model}.csv")
    quality_report.get_details(property_name='Column Pair Trends').to_csv(f"{out_directory}/column_pair_trends_{prefix}_{model}.csv")
    quality_report.save(f"{out_directory}/quality_report_{prefix}_{model}.pkl")
    print(quality_report)

    # Do a diagnostic
    diagnostic_report = run_diagnostic(
        real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
    ) #train
    diagnostic_report.get_details(property_name='Coverage').to_csv(f"{out_directory}/coverage_{prefix}_{model}.csv")
    diagnostic_report.get_details(property_name='Boundary').to_csv(f"{out_directory}/boundary_{prefix}_{model}.csv")
    diagnostic_report.get_details(property_name='Synthesis').to_csv(f"{out_directory}/synthesis_{prefix}_{model}.csv")
    # diagnostic_report.get_details(property_name='Data Validity').to_csv(f"{out_directory}/validity_{prefix}_{model}.csv")
    # diagnostic_report.get_details(property_name='Data Structure').to_csv(f"{out_directory}/validity_{prefix}_{model}.csv")
    diagnostic_report.save(f"{out_directory}/diagnostic_report_{prefix}_{model}.pkl")
    print(diagnostic_report)

    # Additional metrics
    scores = {"column": [], "range_coverage": [], "category_coverage": [], "statistic_similarity": []}
    for col in real_data.columns:
        col_dtype = metadata.columns[col]['sdtype']
        scores["column"].append(col)
        range_coverage = None
        category_coverage = None
        statistic_similarity = None

        try:
            if col_dtype == "categorical":
                category_coverage = CategoryCoverage.compute(real_data=real_data[col], synthetic_data=synthetic_data[col])
            else:
                range_coverage = RangeCoverage.compute(real_data=real_data[col], synthetic_data=synthetic_data[col])
                statistic_similarity = StatisticSimilarity.compute(real_data=real_data[col], synthetic_data=synthetic_data[col])
        except:
            pass
        scores["range_coverage"].append(range_coverage)
        scores["category_coverage"].append(category_coverage)
        scores["statistic_similarity"].append(statistic_similarity)

    metrics_df = pd.DataFrame(scores)

    metrics_df.to_csv(f"{out_directory}/other_report_{prefix}_{model}.csv")

def generate_synthetic_data(
    data, model, 
    index='index',
    date_columns=[],
    model_params=None,
    num_rows=1000, 
    random_seed=42, 
    prefix='', 
    out_directory=os.path.dirname(os.path.realpath(__file__)),
    split=True,
    save_training_data=False,
    verbose=False,
    evaluate=False
):
    """
    This function takes the training data and augments it
    :param data: data to be augemented
    :param model: string of the model to be used to augment the data, 
    options are fast_ml, gaussian, ct_gan, copula_gan, or tvae
    :param model_params: dictionary of the hyperparameters to be used for the model, default parameters used if None
    :param num_rows: number of rows in the augmented dataset
    :param data_type: title for csv
    :param random_seed: random seed for data split
    :param save: boolean, whether to save the generated data to csv
    :return: synthetic training data and test data
    """

    metadata = SingleTableMetadata()

    metadata.detect_from_dataframe(data=data)

    metadata.update_column(column_name=index, sdtype="id")

    metadata.set_primary_key(index)

    metadata.validate()

    if split:
        train = data.sample(frac=0.7, random_state=random_seed)  # TODO: to parametrize the train/test ratio
        sample_for_testing = data.drop(train.index)
    else:
        train = data


    if model == "gaussian":
        if model_params is None:
            synthesizer = GaussianCopulaSynthesizer(
                metadata,  
                enforce_min_max_values=True,
                enforce_rounding=True,
                default_distribution="norm",
            )
        else:
            synthesizer = GaussianCopulaSynthesizer(**model_params)
    elif model == "ct_gan":
        if model_params is None:
            synthesizer = CTGANSynthesizer(
                metadata, 
                enforce_rounding=True, 
                epochs=5000,
            )
        else:
            synthesizer = CTGANSynthesizer(**model_params)

    elif model == "copula_gan":
        if model_params is None:
            synthesizer = CopulaGANSynthesizer(
                metadata,  
                enforce_min_max_values=True,
                enforce_rounding=True,
                default_distribution="norm",
                epochs=5000,
            )
        else:
            synthesizer = CopulaGANSynthesizer(**model_params)

    elif model == "tvae":
        if model_params is None:
            synthesizer = TVAESynthesizer(
                metadata,  
                enforce_min_max_values=True,
                enforce_rounding=True,
                epochs=5000,
                loss_factor=2,
            )
        else:
            synthesizer = TVAESynthesizer(**model_params)

    else:
        raise ValueError(
            "incorrect model, please input gaussian, ct_gan, copula_gan, or tvae"
        )

    if verbose:
        print('Fitting synthesizer...')
    synthesizer.fit(data=train)

    if verbose:
        print('Generating synthetic data...')
    synthetic_data = synthesizer.sample(num_rows=num_rows)

    os.makedirs(f"{out_directory}/models/", exist_ok=True)

    if verbose:
        print('Saving model...')
    synthesizer.save(f"{out_directory}/models/{model}_{prefix}.pkl")


    if evaluate:
        if verbose:
            print('Evaluating synthetic data...')
        evaluate_synthetic_data(synthetic_data, train, metadata, prefix, out_directory, model)

    # Remove duplicate rows in synthetic data
    synthetic_data = synthetic_data.drop_duplicates()

    # Remove rows in synthetic data that are identical to rows in real data
    isin_real_data = synthetic_data.isin(data.to_dict(orient='list')).all(axis=1)
    synthetic_data = synthetic_data[~isin_real_data]

    synthetic_data = clean_patient_data(synthetic_data, date_columns)

    if save_training_data:
        metadata.save_to_json(filepath=f'{out_directory}/{prefix}_metadata.json')
        train.to_csv(f"{out_directory}/{prefix}_{random_seed}_train_samples.csv")
        sample_for_testing.to_csv(f"{out_directory}/{prefix}_{random_seed}_test_samples.csv")
    
    synthetic_data.to_csv(f"{out_directory}/{prefix}_{model}_augmented_samples.csv")

    if split:
        return synthetic_data, sample_for_testing

    else:
        return synthetic_data
