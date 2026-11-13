from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator
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
import re

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/fraud_db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# DATABASE MODELS
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
    card_network = Column(String)
    card_number = Column(String)
    amount = Column(Float)
    hour_of_day = Column(Integer)
    user_age = Column(Integer)
    is_international = Column(Integer)
    ip_address = Column(String) # NEW
    device_id = Column(String)  # NEW
    prediction = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    card_network = Column(String)
    card_number = Column(String)
    amount = Column(Float)
    prediction = Column(String)
    probability = Column(Float)
    model_version = Column(String)
    risk_factors = Column(String)

Base.metadata.create_all(bind=engine)

SECRET_KEY = "super_secret_bank_key_123"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def verify_password(plain_password, hashed_password): return pwd_context.verify(plain_password, hashed_password)
def get_password_hash(password): return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"username": payload.get("sub"), "role": payload.get("role")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin": raise HTTPException(status_code=403, detail="Admin only.")
    return current_user

MODEL_VERSION = "1.0.0"
def train_model():
    df = pd.read_csv("transactions.csv").sample(n=100000, random_state=42)
    X = df[['amount', 'hour_of_day', 'user_age', 'is_international']]
    y = df['is_fraud']
    m = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss')
    m.fit(X, y)
    return m
model = train_model()

def is_valid_luhn(card_number: str) -> bool:
    clean_number = re.sub(r'[\s-]', '', card_number)
    if not clean_number.isdigit(): return False
    digits = [int(d) for d in clean_number]
    checksum = 0
    for i, digit in enumerate(digits[::-1]):
        if i % 2 == 1:
            digit *= 2
            if digit > 9: digit -= 9
        checksum += digit
    return checksum % 10 == 0

def detect_card_network(card_number: str) -> str:
    clean_number = re.sub(r'[\s-]', '', card_number)
    if re.match(r"^4", clean_number): return "Visa"
    if re.match(r"^(5[1-5]|2[2-7])", clean_number): return "Mastercard"
    if re.match(r"^3[47]", clean_number): return "American Express"
    if re.match(r"^62", clean_number): return "UnionPay"
    return "Unknown"

# ==========================================
# SCHEMAS
# ==========================================
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=8)
    @field_validator('username')
    def validate_username(cls, v):
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", v): raise ValueError('Invalid username.')
        return v

class TransactionInput(BaseModel):
    card_network: str
    card_number: str = Field(..., min_length=16, max_length=19)
    amount: float = Field(..., gt=0)
    hour_of_day: int = Field(..., ge=0, le=23)
    user_age: int = Field(..., ge=18, le=120)
    is_international: int = Field(..., ge=0, le=1)
    ip_address: str # NEW
    device_id: str  # NEW
    @field_validator('card_number')
    def validate_card_number(cls, v: str) -> str:
        if not is_valid_luhn(v): raise ValueError('Invalid card number.')
        return v

class PredictionResponse(BaseModel):
    prediction: str
    fraud_probability: float
    timestamp: datetime
    risk_factors: Dict[str, float]

class HistoryItem(BaseModel):
    card_network: str
    card_number: str
    amount: float
    hour_of_day: int
    user_age: int
    ip_address: str
    device_id: str
    prediction: str
    timestamp: datetime
    class Config: from_attributes = True

class AuditLogResponse(BaseModel):
    timestamp: datetime
    card_network: str
    card_number: str
    amount: float
    prediction: str
    probability: float
    model_version: str
    risk_factors: str
    class Config: from_attributes = True

app = FastAPI(title="Bank Fraud Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(tx: TransactionInput, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    detected_network = detect_card_network(tx.card_number)
    
    # 1. Velocity Check (Rule)
    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    recent_txs = db.query(Transaction).filter(Transaction.card_number == tx.card_number, Transaction.timestamp >= one_minute_ago).count()

    # 2. Impossible Travel Check (Rule)
    impossible_travel = False
    last_tx = db.query(Transaction).filter(Transaction.card_number == tx.card_number).order_by(Transaction.timestamp.desc()).first()
    if last_tx and last_tx.ip_address != tx.ip_address:
        if last_tx.is_international != tx.is_international:
            impossible_travel = True # Simple simulation: local -> intl in short time

    # 3. ML Model
    input_data = pd.DataFrame([{'amount': tx.amount, 'hour_of_day': tx.hour_of_day, 'user_age': tx.user_age, 'is_international': tx.is_international}])
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    
    # 4. Three-Tier Risk-Based Authentication (RBA)
    if probability > 0.85 or recent_txs >= 3 or impossible_travel:
        result = "BLOCK"
    elif probability > 0.40:
        result = "STEP-UP AUTH" # Requires OTP
    else:
        result = "APPROVE"

    # 5. Explainability
    importances = model.feature_importances_
    feature_names = ['amount', 'hour_of_day', 'user_age', 'is_international']
    explanations = {feature_names[i]: round(float(importances[i] * 10), 2) for i in range(len(feature_names)) if importances[i] > 0.05}
    top_reasons = dict(sorted(explanations.items(), key=lambda item: item[1], reverse=True)[:2])
    
    if recent_txs >= 3: top_reasons = {"Velocity Rule": 10.0, **top_reasons}
    if impossible_travel: top_reasons = {"Impossible Travel (Geo mismatch)": 10.0, **top_reasons}

    # 6. Save
    new_tx = Transaction(
        card_network=detected_network, card_number=tx.card_number, amount=tx.amount, 
        hour_of_day=tx.hour_of_day, user_age=tx.user_age, is_international=tx.is_international, 
        ip_address=tx.ip_address, device_id=tx.device_id, prediction=result
    )
    db.add(new_tx)
    db.add(AuditLog(card_network=detected_network, card_number=tx.card_number, amount=tx.amount, prediction=result, probability=round(probability * 100, 2), model_version=MODEL_VERSION, risk_factors=json.dumps(top_reasons)))
    db.commit()
    db.refresh(new_tx)
    
    return {"prediction": result, "fraud_probability": round(probability * 100, 2), "timestamp": new_tx.timestamp, "risk_factors": top_reasons}

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first(): raise HTTPException(status_code=400, detail="Exists")
    role = "admin" if user.username.lower() == "admin" else "teller"
    db.add(User(username=user.username, hashed_password=get_password_hash(user.password), role=role))
    db.commit()
    return {"message": f"User created as {role}"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password): raise HTTPException(status_code=401, detail="Bad credentials")
    token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer", "role": user.role}

@app.get("/history", response_model=List[HistoryItem])
def get_history(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    return db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(10).all()

@app.get("/admin/audit", response_model=List[AuditLogResponse])
def get_audit_logs(db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin_user)):
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(10).all()

@app.post("/admin/retrain")
def retrain_model(current_user: dict = Depends(get_current_admin_user)):
    global model, MODEL_VERSION
    model = train_model()
    MODEL_VERSION = f"1.0.{random.randint(1, 9)}"
    return {"message": f"Model retrained. Version: {MODEL_VERSION}"}

@app.get("/admin/drift")
def check_drift(db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin_user)):
    recent_txs = db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(100).all()
    if len(recent_txs) < 10: return {"status": "Not enough data", "message": "Need at least 10 transactions."}
    recent_avg = sum([tx.amount for tx in recent_txs]) / len(recent_txs)
    drift = recent_avg > 300.0
    return {"drift_detected": drift, "recent_average_amount": round(recent_avg, 2), "baseline_average_amount": 150.0, "recent_fraud_rate_percentage": round((sum([1 for tx in recent_txs if tx.prediction == "BLOCK"]) / len(recent_txs)) * 100, 2), "message": "🚨 Drift Detected" if drift else "✅ Healthy"}