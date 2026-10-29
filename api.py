from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Fraud Detection API")

# --- 1. Load Data and Train Model on Startup ---
print("Loading data and training model...")
# We load a sample of the data to train quickly on startup
df = pd.read_csv("transactions.csv").sample(n=100000, random_state=42)
X = df[['amount', 'hour_of_day', 'user_age', 'is_international']]
y = df['is_fraud']

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
print("Model trained and ready for inference!")

# --- 2. Define the Input Schema ---
class Transaction(BaseModel):
    amount: float
    hour_of_day: int
    user_age: int
    is_international: int

# --- 3. The Prediction Endpoint ---
@app.post("/predict")
def predict_fraud(tx: Transaction):
    # Convert input to a DataFrame to match feature names
    input_data = pd.DataFrame([{
        'amount': tx.amount,
        'hour_of_day': tx.hour_of_day,
        'user_age': tx.user_age,
        'is_international': tx.is_international
    }])
    
    # Make prediction
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    
    result = "FRAUD DETECTED" if prediction == 1 else "SAFE"
    
    return {
        "prediction": result,
        "fraud_probability": round(probability * 100, 2),
        "transaction_details": tx
    }

@app.get("/")
def health_check():
    return {"status": "API is running"}