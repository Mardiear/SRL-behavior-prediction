# src/experiments/plot_results.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_mse_by_model(df):
    """
    Plots MSE by model.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='mse', data=df)
    plt.title('MSE by Model')
    plt.xlabel('Model')
    plt.ylabel('MSE')
    plt.savefig('../results/figures/mse_by_model.png')
    plt.show()

def plot_mae_by_model(df):
    """
    Plots MAE by model.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='mae', data=df)
    plt.title('MAE by Model')
    plt.xlabel('Model')
    plt.ylabel('MAE')
    plt.savefig('../results/figures/mae_by_model.png')
    plt.show()

if __name__ == "__main__":
    # Load results
    results_df = pd.read_csv('../results/metrics/all_results.csv')

    # Plot results
    plot_mse_by_model(results_df)
    plot_mae_by_model(results_df)