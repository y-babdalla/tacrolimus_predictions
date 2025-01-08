"""Functions to generate and evaluate synthetic data using various SDV synthesiser models."""

import logging
import os

import pandas as pd
from sdmetrics.single_column import CategoryCoverage, RangeCoverage, StatisticSimilarity
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CopulaGANSynthesizer,
    CTGANSynthesizer,
    GaussianCopulaSynthesizer,
    TVAESynthesizer,
)

from scripts.utils.clean_patient_data import clean_patient_data


def evaluate_synthetic_data(
    synthetic_data: pd.DataFrame,
    real_data: pd.DataFrame,
    metadata: SingleTableMetadata,
    prefix: str,
    out_directory: str,
    model: str,
) -> None:
    """Evaluate the quality of synthetic data by generating a quality report and diagnostic report.

    In addition, computes extra metrics such as range coverage, category coverage,
    and statistic similarity for each column.

    Args:
        synthetic_data (pd.DataFrame):
            The synthetic dataset to evaluate.
        real_data (pd.DataFrame):
            The original (real) dataset used to train the synthesiser.
        metadata (SingleTableMetadata):
            SDV metadata describing the table structure.
        prefix (str):
            A string prefix appended to output file names.
        out_directory (str):
            Directory path to save the evaluation reports.
        model (str):
            Identifier of the synthetic model (e.g., 'gaussian', 'ct_gan').
    """
    # Evaluate quality
    quality_report = evaluate_quality(real_data, synthetic_data, metadata)
    os.makedirs(f"{out_directory}/", exist_ok=True)
    quality_report.get_details(property_name="Column Shapes").to_csv(
        f"{out_directory}/column_shapes_{prefix}_{model}.csv"
    )
    quality_report.get_details(property_name="Column Pair Trends").to_csv(
        f"{out_directory}/column_pair_trends_{prefix}_{model}.csv"
    )
    quality_report.save(f"{out_directory}/quality_report_{prefix}_{model}.pkl")
    logging.info(quality_report)

    diagnostic_report = run_diagnostic(
        real_data=real_data, synthetic_data=synthetic_data, metadata=metadata
    )
    diagnostic_report.get_details(property_name="Coverage").to_csv(
        f"{out_directory}/coverage_{prefix}_{model}.csv"
    )
    diagnostic_report.get_details(property_name="Boundary").to_csv(
        f"{out_directory}/boundary_{prefix}_{model}.csv"
    )
    diagnostic_report.get_details(property_name="Synthesis").to_csv(
        f"{out_directory}/synthesis_{prefix}_{model}.csv"
    )

    diagnostic_report.save(f"{out_directory}/diagnostic_report_{prefix}_{model}.pkl")
    logging.info(diagnostic_report)

    # Compute additional single-column metrics
    scores = {
        "column": [],
        "range_coverage": [],
        "category_coverage": [],
        "statistic_similarity": [],
    }

    for col in real_data.columns:
        col_dtype = metadata.columns[col]["sdtype"]
        scores["column"].append(col)
        range_coverage = None
        category_coverage = None
        statistic_similarity = None

        try:
            if col_dtype == "categorical":
                category_coverage = CategoryCoverage.compute(
                    real_data=real_data[col], synthetic_data=synthetic_data[col]
                )
            else:
                range_coverage = RangeCoverage.compute(
                    real_data=real_data[col], synthetic_data=synthetic_data[col]
                )
                statistic_similarity = StatisticSimilarity.compute(
                    real_data=real_data[col], synthetic_data=synthetic_data[col]
                )
        except Exception:
            pass

        scores["range_coverage"].append(range_coverage)
        scores["category_coverage"].append(category_coverage)
        scores["statistic_similarity"].append(statistic_similarity)

    metrics_df = pd.DataFrame(scores)
    metrics_df.to_csv(f"{out_directory}/other_report_{prefix}_{model}.csv")


def generate_synthetic_data(  # noqa: C901, PLR0912
    data: pd.DataFrame,
    model: str,
    index: str = "index",
    date_columns: list[str] | None = None,
    model_params: dict[str, int | float | str] | None = None,
    num_rows: int = 1000,
    random_seed: int = 42,
    prefix: str = "",
    out_directory: str = os.path.dirname(os.path.realpath(__file__)),
    split: bool = True,
    save_training_data: bool = False,
    verbose: bool = False,
    evaluate: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data using an SDV synthesiser model (GaussianCopula, CTGAN, CopulaGAN, or TVAE).

    Allows for optional train/test splitting, as well as saving, cleaning, and evaluating
    the generated data.

    Args:
        data (pd.DataFrame):
            Input dataset for training the synthesiser.
        model (str):
            String identifier of the SDV model. Should be one of:
            {'gaussian', 'ct_gan', 'copula_gan', 'tvae'}.
        index (str):
            Name of the column to treat as an ID field. Defaults to "index".
        date_columns (list[str] | None):
            Names of date columns to be cleaned in the synthetic data. Defaults to None.
        model_params (dict[str, int | float | str] | None):
            Hyperparameters for the chosen model. Uses default parameters if None.
        num_rows (int):
            Number of rows to generate in the synthetic dataset.
        random_seed (int):
            Seed for random sampling and reproducibility.
        prefix (str):
            String prefix appended to output files.
        out_directory (str):
            Directory path where model and data files will be saved.
        split (bool):
            If True, splits the input data into training and testing sets before fitting.
        save_training_data (bool):
            If True, saves the training and testing sets to CSV, and also saves the metadata file.
        verbose (bool):
            If True, prints progress messages to the console.
        evaluate (bool):
            If True, runs the `evaluate_synthetic_data` function on the generated data.

    Returns:
        pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]:
            - If `split` is False, returns the synthetic data as a DataFrame.
            - If `split` is True, returns a tuple of (synthetic_data, sample_for_testing).
    """
    if date_columns is None:
        date_columns = []

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)

    metadata.update_column(column_name=index, sdtype="id")
    metadata.set_primary_key(index)
    metadata.validate()

    if split:
        train = data.sample(frac=0.7, random_state=random_seed)
        sample_for_testing = data.drop(train.index)
    else:
        train = data

    if model == "gaussian":
        if model_params is None:
            synthesizer = GaussianCopulaSynthesizer(
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                default_distribution="norm",
            )
        else:
            synthesizer = GaussianCopulaSynthesizer(**model_params)
    elif model == "ct_gan":
        if model_params is None:
            synthesizer = CTGANSynthesizer(metadata=metadata, enforce_rounding=True, epochs=5000)
        else:
            synthesizer = CTGANSynthesizer(**model_params)
    elif model == "copula_gan":
        if model_params is None:
            synthesizer = CopulaGANSynthesizer(
                metadata=metadata,
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
                metadata=metadata,
                enforce_min_max_values=True,
                enforce_rounding=True,
                epochs=5000,
                loss_factor=2,
            )
        else:
            synthesizer = TVAESynthesizer(**model_params)
    else:
        raise ValueError("Model must be one of: 'gaussian', 'ct_gan', 'copula_gan', or 'tvae'.")

    if verbose:
        logging.info("Fitting synthesiser...")
    synthesizer.fit(data=train)

    if verbose:
        logging.info("Generating synthetic data...")
    synthetic_data = synthesizer.sample(num_rows=num_rows)

    os.makedirs(f"{out_directory}/models/", exist_ok=True)
    if verbose:
        logging.info("Saving synthesiser model...")
    synthesizer.save(f"{out_directory}/models/{model}_{prefix}.pkl")

    if evaluate:
        if verbose:
            logging.info("Evaluating synthetic data...")
        evaluate_synthetic_data(
            synthetic_data=synthetic_data,
            real_data=train,
            metadata=metadata,
            prefix=prefix,
            out_directory=out_directory,
            model=model,
        )

    # Remove duplicates and rows matching real data to prevent data leakage
    synthetic_data = synthetic_data.drop_duplicates()
    isin_real_data = synthetic_data.isin(data.to_dict(orient="list")).all(axis=1)
    synthetic_data = synthetic_data[~isin_real_data]

    synthetic_data = clean_patient_data(synthetic_data, date_columns)

    if save_training_data:
        metadata.save_to_json(filepath=f"{out_directory}/{prefix}_metadata.json")
        train.to_csv(f"{out_directory}/{prefix}_{random_seed}_train_samples.csv", index=False)
        if split:
            sample_for_testing.to_csv(
                f"{out_directory}/{prefix}_{random_seed}_test_samples.csv", index=False
            )

    synthetic_data.to_csv(f"{out_directory}/{prefix}_{model}_augmented_samples.csv", index=False)

    if split:
        return synthetic_data, sample_for_testing
    return synthetic_data
