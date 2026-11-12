from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import pandas as pd
import os
import xgboost as xgb

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
        username = payload.get("sub")
        role = payload.get("role")
        if username is None or role is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions. Admin only.")
    return current_user

# ==========================================
# 4. MACHINE LEARNING MODEL SETUP (XGBoost)
# ==========================================
print("Loading data and training XGBoost model...")
try:
    df = pd.read_csv("transactions.csv").sample(n=100000, random_state=42)
    X = df[['amount', 'hour_of_day', 'user_age', 'is_international']]
    y = df['is_fraud']

    model = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42, 
        eval_metric='logloss'
    )
    model.fit(X, y)
    print("Model trained and ready for inference!")
except Exception as e:
    print(f"Error loading model: {e}")

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

# ==========================================
# 6. FASTAPI APP & MIDDLEWARE
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

@app.get("/")
def health_check():
    return {"status": "Bank API is running securely"}

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = get_password_hash(user.password)
    assigned_role = "admin" if user.username.lower() == "admin" else "teller"
    
    new_user = User(username=user.username, hashed_password=hashed_pw, role=assigned_role)
    db.add(new_user)
    db.commit()
    return {"message": f"User created successfully as {assigned_role}"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(
    tx: TransactionInput, 
    db: Session = Depends(get_db), 
    current_user: dict = Depends(get_current_user)
):
    # --- 1. RULE ENGINE: VELOCITY CHECK ---
    one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
    recent_txs = db.query(Transaction).filter(
        Transaction.card_number == tx.card_number,
        Transaction.timestamp >= one_minute_ago
    ).count()

    # --- 2. MACHINE LEARNING MODEL ---
    input_data = pd.DataFrame([{
        'amount': tx.amount,
        'hour_of_day': tx.hour_of_day,
        'user_age': tx.user_age,
        'is_international': tx.is_international
    }])
    
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    result = "FRAUD DETECTED" if prediction == 1 else "SAFE"
    
    # --- 3. EXPLAINABILITY ---
    importances = model.feature_importances_
    feature_names = ['amount', 'hour_of_day', 'user_age', 'is_international']
    
    explanations = {}
    for i, feature in enumerate(feature_names):
        if importances[i] > 0.05: 
            explanations[feature] = round(float(importances[i] * 10), 2) 
            
    top_reasons = dict(sorted(explanations.items(), key=lambda item: item[1], reverse=True)[:2])
    
    # --- 4. OVERRIDE WITH RULE ENGINE ---
    if recent_txs >= 3:
        result = "FRAUD DETECTED"
        probability = 99.9
        top_reasons = {"Velocity Rule (3+ tx/min)": 10.0, **top_reasons}

    # --- 5. SAVE TO DATABASE ---
    new_tx = Transaction(
        card_number=tx.card_number,
        amount=tx.amount,
        hour_of_day=tx.hour_of_day,
        user_age=tx.user_age,
        is_international=tx.is_international,
        prediction=result
    )
    db.add(new_tx)
    db.commit()
    db.refresh(new_tx)
    
    return {
        "prediction": result,
        "fraud_probability": round(probability * 100, 2),
        "timestamp": new_tx.timestamp,
        "risk_factors": top_reasons
    }

@app.get("/history", response_model=List[HistoryItem])
def get_history(db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)):
    transactions = db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(10).all()
    return transactions

@app.get("/admin/users")
def get_all_users(db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin_user)):
    users = db.query(User).all()
    return [{"id": u.id, "username": u.username, "role": u.role} for u in users]

# ==========================================
# 8. ADMIN DRIFT MONITORING ENDPOINT
# ==========================================
@app.get("/admin/drift")
def check_drift(db: Session = Depends(get_db), current_user: dict = Depends(get_current_admin_user)):
    # Get the last 100 transactions
    recent_txs = db.query(Transaction).order_by(Transaction.timestamp.desc()).limit(100).all()
    
    if len(recent_txs) < 10:
        return {"status": "Not enough data", "message": "Need at least 10 transactions to analyze drift."}

    # Calculate the average amount of recent transactions
    recent_avg_amount = sum([tx.amount for tx in recent_txs]) / len(recent_txs)
    
    # The training baseline (our synthetic data averaged around $150)
    baseline_avg_amount = 150.0 
    
    # Statistical Drift Check (If recent avg is 2x higher than baseline)
    drift_detected = recent_avg_amount > (baseline_avg_amount * 2)
    
    # Count fraud vs safe in the recent batch
    fraud_count = sum([1 for tx in recent_txs if tx.prediction == "FRAUD DETECTED"])
    fraud_rate = (fraud_count / len(recent_txs)) * 100

    return {
        "drift_detected": drift_detected,
        "recent_average_amount": round(recent_avg_amount, 2),
        "baseline_average_amount": baseline_avg_amount,
        "recent_fraud_rate_percentage": round(fraud_rate, 2),
        "message": "🚨 ALERT: Data drift detected! Fraud patterns may have changed." if drift_detected else "✅ System healthy. Data distribution is normal."
    }