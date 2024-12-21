from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np


def plot_dose_vs_level(model, row_df, dose_range=(0.5, 10), target_level_range=(5, 20)):
    """
    Plots dose vs. predicted level and finds the optimal dose using Bayesian Optimization.

    Parameters:
    - model: Trained predictive model.
    - row_df: DataFrame containing the input features for prediction.
    - dose_range: Tuple containing the start and stop values for dose exploration.
    - target_level_range: Tuple containing the minimum and maximum target levels.

    Returns:
    - optimal_dose: The dose value that optimizes the predicted level within the target range.
    - optimal_level: The predicted level at the optimal dose.
    """

    # Generate a range of dose values to test
    dose_values = np.linspace(start=dose_range[0], stop=dose_range[1], num=22)
    predictions = []
    for dose in dose_values:
        row_df_copy = row_df.copy()
        row_df_copy['dose'] = dose

        predictions.append(model.predict(row_df_copy))

    # Plot dose vs level
    plt.figure(figsize=(8, 6))
    plt.plot(dose_values, predictions, label='Predicted Level', color='b')
    plt.xlabel('Dose')
    plt.ylabel('Level')
    plt.title('Dose vs Level Predictions')
    plt.grid(True)

    # Highlight the target level range if provided
    min_level, max_level = target_level_range
    plt.axhspan(min_level, max_level, color='yellow', alpha=0.3, label='Target Level Range')
    plt.legend()

    # Define the objective function for Bayesian Optimization
    def objective_function(dose):
        # Update dose in row_df
        row_df_copy = row_df.copy()
        row_df_copy['dose'] = dose

        # Predict level
        predicted_level = model.predict(row_df_copy)

        target_level = np.mean([min_level, max_level])

        loss = (predicted_level - target_level) ** 2

        # Return negative loss because the optimizer maximizes the objective function
        return -loss

    # Set up Bayesian Optimization
    pbounds = {'dose': dose_range}
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
    )

    # Run the optimizer
    optimizer.maximize(
        init_points=5,
        n_iter=25,
    )

    # Get the optimal dose
    optimal_dose = optimizer.max['params']['dose']

    # Get the predicted level at the optimal dose
    row_df_copy = row_df.copy()
    row_df_copy['dose'] = optimal_dose
    optimal_level = model.predict(row_df_copy)

    # Plot the optimal dose point with a larger marker and higher z-order
    plt.scatter([optimal_dose], [optimal_level], color='red', label='Optimal Dose', s=100, zorder=5)

    # Add an annotation to label the optimal dose point
    plt.annotate(
        f'Optimal Dose: {optimal_dose:.2f}\nPredicted Level: {optimal_level:.2f}',
        xy=(optimal_dose, optimal_level),
        xytext=(optimal_dose, optimal_level + (max(predictions) - min(predictions)) * 0.05),
        arrowprops=dict(facecolor='red', shrink=0.05),
        fontsize=9,
        color='red',
        ha='center'
    )

    # Add legend and display the plot
    plt.legend()
    plt.show()

    # Print optimal dose and predicted level
    print(f"Optimal Dose: {optimal_dose}")
    print(f"Predicted Level at Optimal Dose: {optimal_level}")

    return optimal_dose, optimal_level
