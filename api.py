from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import pandas as pd
import os
import xgboost as xgb
import json
import random

# ==========================================
# 1. DATABASE SETUP & PSYCOPG2 FIX
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/fraud_db")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 2. DATABASE MODELS
# ==========================================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="teller")

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    card_number = Column(String)
    amount = Column(Float)
    hour_of_day = Column(Integer)
    user_age = Column(Integer)
    is_international = Column(Integer)
    prediction = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

# NEW: Immutable Audit Trail
class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    card_number = Column(String)
    amount = Column(Float)
    prediction = Column(String)
    probability = Column(Float)
    model_version = Column(String)
    risk_factors = Column(String) # Stored as JSON string

Base.metadata.create_all(bind=engine)

# ==========================================
# 3. SECURITY & JWT SETUP
# ==========================================
SECRET_KEY = "super_secret_bank_key_123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload.get("sub"), "role": payload.get("role")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only.")
    return current_user

# ==========================================
# 4. MACHINE LEARNING MODEL (XGBoost)
# ==========================================
MODEL_VERSION = "1.0.0"
print(f"Loading data and training XGBoost model (v{MODEL_VERSION})...")

def train_model():
    df = pd.read_csv("transactions.csv").sample(n=100000, random_state=42)
    X = df[['amount', 'hour_of_day', 'user_age', 'is_international']]
    y = df['is_fraud']
    m = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss')
    m.fit(X, y)
    return m

model = train_model()
print("Model trained and ready!")

# ==========================================
# 5. PYDANTIC SCHEMAS
# ==========================================
class TransactionInput(BaseModel):
    card_number: str
    amount: float
    hour_of_day: int
    user_age: int
    is_international: int

class PredictionResponse(BaseModel):
    prediction: str
    fraud_probability: float
    timestamp: datetime
    risk_factors: Dict[str, float]

class HistoryItem(BaseModel):
    card_number: str
    amount: float
    hour_of_day: int
    user_age: int
    prediction: str
    timestamp: datetime

    class Config:
        from_attributes = True

class UserCreate(BaseModel):
    username: str
    password: str

class AuditLogResponse(BaseModel):
    timestamp: datetime
    card_number: str
    amount: float
    prediction: str
    probability: float
    model_version: str
    risk_factors: str

    class Config:
        from_attributes = True

# ==========================================
# 6. FASTAPI APP
# ==========================================
app = FastAPI(title="Bank Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 7. API ROUTES
# ==========================================

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(tx: TransactionInput, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    # Velocity Check
    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    recent_txs = db.query(Transaction).filter(Transaction.card_number == tx.card_number, Transaction.timestamp >= one_minute_ago).count()

    # ML Model
    input_data = pd.DataFrame([{'amount': tx.amount, 'hour_of_day': tx.hour_of_day, 'user_age': tx.user_age, 'is_international': tx.is_international}])
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    result = "FRAUD DETECTED" if prediction == 1 else "SAFE"
    
    # Explainability
    importances = model.feature_importances_
    feature_names = ['amount', 'hour_of_day', 'user_age', 'is_international']
    explanations = {feature_names[i]: round(float(importances[i] * 10), 2) for i in range(len(feature_names)) if importances[i] > 0.05}
    top_reasons = dict(sorted(explanations.items(), key=lambda item: item[1], reverse=True)[:2])
    
    # Override with Velocity Rule
    if recent_txs >= 3:
        result = "FRAUD DETECTED"
        probability = 99.9
        top_reasons = {"Velocity Rule (3+ tx/min)": 10.0, **top_reasons}

    # Save to standard Transaction Table
    new_tx = Transaction(card_number=tx.card_number, amount=tx.amount, hour_of_day=tx.hour_of_day, user_age=tx.user_age, is_international=tx.is_international, prediction=result)
    db.add(new_tx)
    
    # Save to Immutable Audit Log!
    new_audit = AuditLog(
        card_number=tx.card_number, amount=tx.amount, prediction=result, probability=round(probability * 100, 2),
        model_version=MODEL_VERSION, risk_factors=json.dumps(top_reasons)
    )
    db.add(new_audit)
    
    db.commit()
    db.refresh(new_tx)
    
    return {"prediction": result, "fraud_probability": round(probability * 100, 2), "timestamp": new_tx.timestamp, "risk_factors": top_reasons}

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already registered")
    role = "admin" if user.username.lower() == "admin" else "teller"
    new_user = User(username=user.username, hashed_password=get_password_hash(user.password), role=role)
    db.add(new_user)
    db.commit()
    return {"message": f"User created successfully as {role}"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

@app.get("/history", response_model=List[HistoryItem])
def get_history(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    return db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(10).all()

# ==========================================
# 8. ADMIN ENDPOINTS
# ==========================================
@app.get("/admin/audit", response_model=List[AuditLogResponse])
def get_audit_logs(db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin_user)):
    # Regulators require viewing the immutable log
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(10).all()

@app.post("/admin/retrain")
def retrain_model(current_user: dict = Depends(get_current_admin_user)):
    global model, MODEL_VERSION
    print("Admin triggered retraining...")
    model = train_model() # Re-trains on the CSV
    # Bump the version to prove it updated
    new_version = f"1.0.{random.randint(1, 9)}"
    MODEL_VERSION = new_version
    return {"message": f"Model retrained successfully. New version: {MODEL_VERSION}"}

@app.get("/admin/drift")
def check_drift(db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin_user)):
    recent_txs = db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(100).all()
    if len(recent_txs) < 10:
        return {"status": "Not enough data", "message": "Need at least 10 transactions to analyze drift."}
    recent_avg_amount = sum([tx.amount for tx in recent_txs]) / len(recent_txs)
    drift_detected = recent_avg_amount > 300.0
    fraud_rate = (sum([1 for tx in recent_txs if tx.prediction == "FRAUD DETECTED"]) / len(recent_txs)) * 100
    return {
        "drift_detected": drift_detected, "recent_average_amount": round(recent_avg_amount, 2),
        "baseline_average_amount": 150.0, "recent_fraud_rate_percentage": round(fraud_rate, 2),
        "message": "🚨 ALERT: Data drift detected!" if drift_detected else "✅ System healthy."
    }