# src/experiments/experiments_results.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_df_from_experiments():
    """
    Gathers all experiment results into a single DataFrame.
    """
    results = []
    for model in ['mlp', 'lstm', 'cnn', 'tcn']:
        df = pd.read_csv(f'../results/metrics/{model}_results.csv')
        results.append(df)
    return pd.concat(results, ignore_index=True)

def rank_results_agg_func(df, comparison_col, rank_by_col, agg_func):
    """
    Ranks results based on a comparison column and aggregation function.
    """
    return df.groupby(comparison_col)[rank_by_col].agg(agg_func).sort_values()

def plot_results(df, x_col, y_col, hue_col, title):
    """
    Plots results using a bar plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, hue=hue_col, data=df)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title=hue_col)
    plt.savefig(f'../results/figures/{title}.png')
    plt.show()

if __name__ == "__main__":
    # Generate DataFrame from experiment results
    results_df = generate_df_from_experiments()

    # Rank results by MSE
    ranked_results = rank_results_agg_func(results_df, 'model', 'mse', 'mean')
    print("Ranked Results by MSE:")
    print(ranked_results)

    # Plot results
    plot_results(results_df, x_col='model', y_col='mse', hue_col='strategy', title='Model Performance by Strategy')