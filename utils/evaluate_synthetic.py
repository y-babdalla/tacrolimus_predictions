import pandas as pd
import numpy as np
from sdmetrics.single_table import KSComplement, TVComplement, LogisticDetection
import pandas as pd
from sdv.metrics.tabular import KSComplement, TVComplement, LogisticDetection, ContinuousKLDivergence, DiscreteKLDivergence
from tqdm import tqdm


def evaluate_generated_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> pd.DataFrame:

    discrete_columns = real_data.select_dtypes(include=["object", "category"]).columns.tolist()
    continuous_columns = [col for col in real_data.columns if col not in discrete_columns]

    kld_scores = []
    # Continuous column evaluation using KSComplement
    if continuous_columns:
        ks_test_score = KSComplement.compute(
            real_data[continuous_columns], synthetic_data[continuous_columns]
        )
        # Continuous column evaluation using KL Divergence
        kl_divergence_scores = [
            ContinuousKLDivergence.compute(
                real_data=real_data, synthetic_data=synthetic_data, column_name=col
            )
            for col in continuous_columns
        ]

        kld_scores += kl_divergence_scores

    # Discrete column evaluation using TVComplement
    if discrete_columns:
        tv_test_score = TVComplement.compute(
            real_data[discrete_columns], synthetic_data[discrete_columns]
        )

        # Discrete column evaluation using KL Divergence
        kl_divergence_scores_disc = [
            DiscreteKLDivergence.compute(
                real_data=real_data, synthetic_data=synthetic_data, column_name=col
            )
            for col in discrete_columns
        ]

        kld_scores += kl_divergence_scores_disc


    divergence = np.mean(kld_scores)

    print("###########################################")
    if continuous_columns:
        print("KS Complement:", ks_test_score)
    if discrete_columns:
        print("TV Complement", tv_test_score)
    print("KL Divergence", divergence)
    print("###########################################")


# Example usage
if __name__ == "__main__":

    # Replace with real spanish data path
    real_data = pd.read_csv("data/mimic/followup_final_v4.csv")

    # Replace with synthetic spanish data path
    synthetic_data = pd.read_csv("data/mimic_gen/mimic_full_gaussian.csv")

    evaluate_generated_data(real_data, synthetic_data)


