# correlation_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load your cleaned dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Example columns expected: ['State', 'Quarter', 'Price_Change', 'Population_Growth', 'Dwelling_Approvals']

states = df['State'].unique()

for state in states:
    print(f"\n--- Correlation Analysis for {state} ---")
    state_data = df[df['State'] == state]

    # Population Growth vs. Price Change
    r_pop, p_pop = pearsonr(state_data['Population_Growth'], state_data['Price_Change'])
    print(f"Population Growth vs. Price Change: r = {r_pop:.2f}, p = {p_pop:.3f}")

    # Dwelling Approvals vs. Price Change
    r_dwell, p_dwell = pearsonr(state_data['Dwelling_Approvals'], state_data['Price_Change'])
    print(f"Dwelling Approvals vs. Price Change: r = {r_dwell:.2f}, p = {p_dwell:.3f}")

    # Optional: Heatmap
    corr_matrix = state_data[['Price_Change', 'Population_Growth', 'Dwelling_Approvals']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix - {state}')
    plt.show()
