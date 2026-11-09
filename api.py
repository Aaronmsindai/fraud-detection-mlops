from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import pandas as pd
import os

# --- 1. Database Setup ---
# THE DATABASE URL WILL BE READ FROM RENDER'S ENVIRONMENT VARIABLES
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/fraud_db")

# Fix for Render's postgres:// vs postgresql:// issue
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 2. Database Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float)
    hour_of_day = Column(Integer)
    user_age = Column(Integer)
    is_international = Column(Integer)
    prediction = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- 3. Security & JWT Setup ---
SECRET_KEY = "super_secret_bank_key_123" # In production, use os.getenv()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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

# --- 4. Load ML Model ---
print("Loading data and training model...")
df = pd.read_csv("transactions.csv").sample(n=100000, random_state=42)
X = df[['amount', 'hour_of_day', 'user_age', 'is_international']]
y = df['is_fraud']

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)
print("Model trained and ready for inference!")

# --- 5. Pydantic Schemas ---
class TransactionInput(BaseModel):
    amount: float
    hour_of_day: int
    user_age: int
    is_international: int

class TransactionResponse(BaseModel):
    prediction: str
    fraud_probability: float
    timestamp: datetime

    class Config:
        orm_mode = True

class UserCreate(BaseModel):
    username: str
    password: str

# --- 6. FastAPI App ---
app = FastAPI(title="Bank Fraud Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 7. Routes ---

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", response_model=TransactionResponse)
def predict_fraud(tx: TransactionInput, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    # 1. Verify the token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # 2. Make prediction
    input_data = pd.DataFrame([{
        'amount': tx.amount,
        'hour_of_day': tx.hour_of_day,
        'user_age': tx.user_age,
        'is_international': tx.is_international
    }])
    
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])
    result = "FRAUD DETECTED" if prediction == 1 else "SAFE"
    
    # 3. Save to Database!
    new_tx = Transaction(
        amount=tx.amount,
        hour_of_day=tx.hour_of_day,
        user_age=tx.user_age,
        is_international=tx.is_international,
        prediction=result
    )
    db.add(new_tx)
    db.commit()
    
    return {
        "prediction": result,
        "fraud_probability": round(probability * 100, 2),
        "timestamp": new_tx.timestamp
    }

@app.get("/")
def health_check():
    return {"status": "Bank API is running securely"}