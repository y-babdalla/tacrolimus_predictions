"""Script to plot the relationship between dose and predicted tacrolimus levels."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization


def plot_dose_vs_level(
    model: Any,
    row_df: pd.DataFrame,
    dose_range: tuple[float, float] = (0.5, 10),
    target_level_range: tuple[float, float] = (5, 20),
) -> tuple[float, float]:
    """Plot predicted dose-level relationships and find an optimal dose via Bayesian Optimisation.

    The function takes a predictive model and a single-row DataFrame, then:
    1. Plots predicted tacrolimus levels over a specified dose range.
    2. Uses Bayesian Optimisation to find a dose that brings the predicted level within a
       user-specified target range.

    Args:
        model:
            A trained predictive model implementing `.predict()`, which expects a DataFrame
            with a 'dose' column among its features.
        row_df:
            A pandas DataFrame representing a single observation (row) of input features.
            The function modifies the 'dose' column of this DataFrame to evaluate different doses.
        dose_range (tuple[float, float], optional):
            The minimum and maximum dose to consider when searching for the optimal dose.
            Defaults to (0.5, 10).
        target_level_range (tuple[float, float], optional):
            The desired lower and upper bounds for the target tacrolimus level.
            Defaults to (5, 20).

    Returns:
        tuple[float, float]:
            - The optimal dose value found by Bayesian Optimisation.
            - The predicted level at the optimal dose.
    """
    dose_values = np.linspace(start=dose_range[0], stop=dose_range[1], num=22)
    predictions = []
    for dose in dose_values:
        row_copy = row_df.copy()
        row_copy["dose"] = dose
        predictions.append(model.predict(row_copy))

    plt.figure(figsize=(8, 6))
    plt.plot(dose_values, predictions, label="Predicted Level", color="b")
    plt.xlabel("Dose")
    plt.ylabel("Level")
    plt.title("Dose vs. Level Predictions")
    plt.grid(True)

    min_level, max_level = target_level_range
    plt.axhspan(min_level, max_level, color="yellow", alpha=0.3, label="Target Level Range")

    def objective_function(_dose: float) -> float:
        """Compute a negative squared-error loss relative to the midpoint of the target."""
        _row_copy = row_df.copy()
        _row_copy["dose"] = _dose
        predicted_level = model.predict(_row_copy)

        target_level = np.mean([min_level, max_level])
        loss = (predicted_level - target_level) ** 2  # MSE w.r.t. target_level

        return -loss

    pbounds = {"dose": dose_range}
    optimizer = BayesianOptimization(f=objective_function, pbounds=pbounds, random_state=42)

    optimizer.maximize(init_points=5, n_iter=25)

    optimal_dose = optimizer.max["params"]["dose"]

    row_optimal = row_df.copy()
    row_optimal["dose"] = optimal_dose
    optimal_level = model.predict(row_optimal)

    plt.scatter(
        [optimal_dose], [optimal_level], color="red", label="Optimal Dose", s=100, zorder=5
    )

    plt.annotate(
        f"Optimal Dose: {optimal_dose:.2f}\nPredicted Level: {optimal_level:.2f}",
        xy=(optimal_dose, optimal_level),
        xytext=(optimal_dose, optimal_level + (max(predictions) - min(predictions)) * 0.05),
        arrowprops=dict(facecolor="red", shrink=0.05),
        fontsize=9,
        color="red",
        ha="center",
    )

    plt.legend()
    plt.show()

    logging.info(f"Optimal Dose: {optimal_dose}")
    logging.info(f"Predicted Level at Optimal Dose: {optimal_level}")

    return optimal_dose, optimal_level
