"""Script to evaluate synthetic data using KS Complement, TV Complement, and KL Divergence."""

import logging

import numpy as np
import pandas as pd
from sdv.metrics.tabular import (
    ContinuousKLDivergence,
    DiscreteKLDivergence,
    KSComplement,
    TVComplement,
)


def evaluate_generated_data(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
    """Evaluate synthetic data using KS Complement, TV Complement, and KL Divergence.

    Args:
        real_data (pd.DataFrame): The real data.
        synthetic_data (pd.DataFrame): The synthetic data.
    """
    discrete_columns = real_data.select_dtypes(include=["object", "category"]).columns.tolist()
    continuous_columns = [col for col in real_data.columns if col not in discrete_columns]
    ks_test_score = None
    tv_test_score = None
    kld_scores = []

    if continuous_columns:
        ks_test_score = KSComplement.compute(
            real_data[continuous_columns], synthetic_data[continuous_columns]
        )
        kl_divergence_scores = [
            ContinuousKLDivergence.compute(
                real_data=real_data, synthetic_data=synthetic_data, column_name=col
            )
            for col in continuous_columns
        ]

        kld_scores += kl_divergence_scores

    if discrete_columns:
        tv_test_score = TVComplement.compute(
            real_data[discrete_columns], synthetic_data[discrete_columns]
        )

        kl_divergence_scores_disc = [
            DiscreteKLDivergence.compute(
                real_data=real_data, synthetic_data=synthetic_data, column_name=col
            )
            for col in discrete_columns
        ]

        kld_scores += kl_divergence_scores_disc

    divergence = np.mean(kld_scores)

    if continuous_columns:
        logging.info(f"KS Complement: {ks_test_score}")
    if discrete_columns:
        logging.info(f"TV Complement: {tv_test_score}")
    logging.info(f"KS Divergence: {divergence}")


if __name__ == "__main__":
    _real_data = pd.read_csv("data/mimic/followup_final_v4.csv")
    _synthetic_data = pd.read_csv("data/mimic_gen/mimic_full_gaussian.csv")

    evaluate_generated_data(_real_data, _synthetic_data)
