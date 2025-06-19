import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 15000

# Simulate features with some realistic variation
avg_balance = np.random.normal(loc=15000, scale=7000, size=n_samples).clip(min=0)
monthly_inflows = np.random.normal(loc=50000, scale=15000, size=n_samples).clip(min=10000)
monthly_outflows = monthly_inflows * np.random.uniform(0.7, 1.2, size=n_samples)

# Generate risk label
risk = ((avg_balance < 10000) & (monthly_outflows > monthly_inflows * 0.9)).astype(int)



data = pd.DataFrame({
    'avg_balance': avg_balance,
    'monthly_inflows': monthly_inflows,
    'monthly_outflows': monthly_outflows,
    'risk': risk
})

high_risk = data[data['risk'] == 1].sample(n=1000, random_state=42)
low_risk = data[data['risk'] == 0].sample(n=4000, random_state=42)

balanced_data = pd.concat([high_risk, low_risk], ignore_index=True).sample(frac=1, random_state=42)

balanced_data.to_csv('data/dataset.csv', index=False)
print("Dataset saved to: data/dataset.csv")