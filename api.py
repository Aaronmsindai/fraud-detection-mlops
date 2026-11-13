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

# ==========================================
# 1. DATABASE SETUP
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
    card_network = Column(String) # NEW: Visa, Mastercard, UnionPay
    card_number = Column(String)
    amount = Column(Float)
    hour_of_day = Column(Integer)
    user_age = Column(Integer)
    is_international = Column(Integer)
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

# ==========================================
# 4. MACHINE LEARNING MODEL
# ==========================================
MODEL_VERSION = "1.0.0"
def train_model():
    df = pd.read_csv("transactions.csv").sample(n=100000, random_state=42)
    X = df[['amount', 'hour_of_day', 'user_age', 'is_international']]
    y = df['is_fraud']
    m = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, eval_metric='logloss')
    m.fit(X, y)
    return m
model = train_model()

# ==========================================
# 5. HELPER FUNCTIONS (VALIDATION)
# ==========================================
def is_valid_luhn(card_number: str) -> bool:
    clean_number = re.sub(r'[\s-]', '', card_number)
    if not clean_number.isdigit(): return False
    digits = [int(d) for d in clean_number]
    checksum = 0
    reverse_digits = digits[::-1]
    for i, digit in enumerate(reverse_digits):
        if i % 2 == 1:
            digit *= 2
            if digit > 9: digit -= 9
        checksum += digit
    return checksum % 10 == 0

# ==========================================
# 6. PYDANTIC SCHEMAS (STRICT VALIDATION)
# ==========================================
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=8)

    @field_validator('username')
    def validate_username(cls, v):
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError('Username must contain only letters, numbers, and underscores.')
        return v

    @field_validator('password')
    def validate_password(cls, v):
        if not re.search(r"[A-Z]", v): raise ValueError("Password must contain an uppercase letter.")
        if not re.search(r"[a-z]", v): raise ValueError("Password must contain a lowercase letter.")
        if not re.search(r"\d", v): raise ValueError("Password must contain a number.")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v): raise ValueError("Password must contain a special character.")
        return v

class TransactionInput(BaseModel):
    card_network: str = Field(..., description="Visa, Mastercard, or UnionPay")
    card_number: str = Field(..., min_length=16, max_length=19)
    amount: float = Field(..., gt=0, lt=1000000)
    hour_of_day: int = Field(..., ge=0, le=23)
    user_age: int = Field(..., ge=18, le=120)
    is_international: int = Field(..., ge=0, le=1)

    @field_validator('card_number')
    def validate_card_number(cls, v: str) -> str:
        if not is_valid_luhn(v):
            raise ValueError('Invalid card number (Luhn check failed).')
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
    prediction: str
    timestamp: datetime

    class Config:
        from_attributes = True

class AuditLogResponse(BaseModel):
    timestamp: datetime
    card_network: str
    card_number: str
    amount: float
    prediction: str
    probability: float
    model_version: str
    risk_factors: str

    class Config:
        from_attributes = True

# ==========================================
# 7. FASTAPI APP & ROUTES
# ==========================================
app = FastAPI(title="Bank Fraud Detection API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(tx: TransactionInput, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    recent_txs = db.query(Transaction).filter(Transaction.card_number == tx.card_number, Transaction.timestamp >= one_minute_ago).count()

    input_data = pd.DataFrame([{'amount': tx.amount, 'hour_of_day': tx.hour_of_day, 'user_age': tx.user_age, 'is_international': tx.is_international}])
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    result = "FRAUD DETECTED" if prediction == 1 else "SAFE"
    
    importances = model.feature_importances_
    feature_names = ['amount', 'hour_of_day', 'user_age', 'is_international']
    explanations = {feature_names[i]: round(float(importances[i] * 10), 2) for i in range(len(feature_names)) if importances[i] > 0.05}
    top_reasons = dict(sorted(explanations.items(), key=lambda item: item[1], reverse=True)[:2])
    
    if recent_txs >= 3:
        result = "FRAUD DETECTED"
        probability = 99.9
        top_reasons = {"Velocity Rule (3+ tx/min)": 10.0, **top_reasons}

    new_tx = Transaction(card_network=tx.card_network, card_number=tx.card_number, amount=tx.amount, hour_of_day=tx.hour_of_day, user_age=tx.user_age, is_international=tx.is_international, prediction=result)
    db.add(new_tx)
    
    new_audit = AuditLog(card_network=tx.card_network, card_number=tx.card_number, amount=tx.amount, prediction=result, probability=round(probability * 100, 2), model_version=MODEL_VERSION, risk_factors=json.dumps(top_reasons))
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

@app.get("/admin/audit", response_model=List[AuditLogResponse])
def get_audit_logs(db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin_user)):
    return db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(10).all()

@app.post("/admin/retrain")
def retrain_model(current_user: dict = Depends(get_current_admin_user)):
    global model, MODEL_VERSION
    model = train_model()
    MODEL_VERSION = f"1.0.{random.randint(1, 9)}"
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