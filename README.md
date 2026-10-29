# 🚨 Real-Time Fraud Detection MLOps Pipeline

An end-to-end Machine Learning pipeline designed to handle highly imbalanced Big Data. This project simulates 1 million bank transactions, processes them using **PySpark**, trains a fraud detection model, and serves real-time predictions via a **FastAPI** microservice deployed on Render.

## 🌐 Live Demo
⚙️ **[Test the Live Fraud Detection API](https://fraud-detection-api-71aw.onrender.com/docs)**
*(Click the link and use the interactive Swagger UI to send a test transaction).*

## 🏗️ Architecture
1. **Data Generation (`generate_data.py`):** Simulates 1,000,000 transactions with a 1% fraud rate. Fraudsters are programmed to have higher amounts, unusual hours (2 AM - 4 AM), and international locations.
2. **Big Data Processing (`process_data.py`):** Uses **PySpark** to load the CSV, assemble features, and train a distributed Random Forest model. 
3. **Backend API (`api.py`):** A **FastAPI** service that loads a trained model and exposes a `/predict` endpoint. It uses Pydantic for strict data validation (e.g., ensuring `age` is a positive integer).
4. **Deployment (`Dockerfile`):** Containerized with Docker and deployed to Render for scalable, real-time inference.

## 🛠️ Tech Stack
- **Big Data:** PySpark
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Backend:** FastAPI, Pydantic, Uvicorn
- **DevOps:** Docker, Render
- **Data Processing:** Pandas, NumPy

## 💻 How to Run Locally

**1. Clone the repository and install dependencies**
```bash
git clone git@github.com:Aaronmsindai/fraud-detection-mlops.git
cd fraud-detection-mlops
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Generate the 1M row dataset

```bash
python generate_data.py
```

3. (Optional) Run the PySpark training script
(Requires Java 17 installed on your system)

```bash
python process_data.py
```

4. Start the FastAPI server

```bash
uvicorn api:app --port 8000
```

(Go to http://localhost:8000/docs to test the API locally).

🔬 Testing the API

You can test the live API using curl:

```bash
curl -X POST https://fraud-detection-api-71aw.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"amount": 4500.00, "hour_of_day": 2, "user_age": 25, "is_international": 1}'
```

Expected Response:

```json
{
  "prediction": "FRAUD DETECTED",
  "fraud_probability": 100.0,
  "transaction_details": {
    "amount": 4500.0,
    "hour_of_day": 2,
    "user_age": 25,
    "is_international": 1
  }
}
```

---

Created by Aaron Msindai

