import pandas as pd
import numpy as np

np.random.seed(42)
n_rows = 1_000_000

print("Generating 1 million transactions...")

data = {
    'transaction_id': np.arange(1, n_rows + 1),
    'amount': np.random.exponential(scale=50, size=n_rows),
    'hour_of_day': np.random.randint(0, 24, size=n_rows),
    'user_age': np.random.randint(18, 80, size=n_rows),
    'is_international': np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05]),
    'is_fraud': np.zeros(n_rows)
}

df = pd.DataFrame(data)

fraud_indices = np.random.choice(df.index, size=int(n_rows * 0.01), replace=False)

df.loc[fraud_indices, 'amount'] = np.random.uniform(500, 5000, size=len(fraud_indices))
df.loc[fraud_indices, 'hour_of_day'] = np.random.choice([0, 1, 2, 3, 4], size=len(fraud_indices))
df.loc[fraud_indices, 'is_international'] = np.random.choice([0, 1], size=len(fraud_indices), p=[0.2, 0.8])
df.loc[fraud_indices, 'is_fraud'] = 1

df.to_csv("transactions.csv", index=False)
print("Data generated successfully! Saved to transactions.csv")
print(f"Fraud rate: {(df['is_fraud'].sum() / n_rows) * 100:.2f}%")
