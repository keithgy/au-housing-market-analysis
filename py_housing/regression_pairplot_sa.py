# regression_pairplot_sa.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load dataset
df = pd.read_csv("data/processed/cleaned_data.csv")

# Filter for South Australia only
df_sa = df[df['State'] == 'SA']

# Select independent and dependent variables
features = ['Population_Growth', 'Dwelling_Approvals', 'State_Final_Demand']
target = 'Price_Change'

# Pairplot to visualize relationships
sns.pairplot(df_sa[features + [target]], kind='reg', plot_kws={'line_kws':{'color':'red'}})
plt.suptitle("SA Housing Market - Pairwise Regression Plots", y=1.02)
plt.show()

# Regression model
X = df_sa[features]
y = df_sa[target]

# Add constant term for intercept
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())
