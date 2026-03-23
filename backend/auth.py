from passlib.context import CryptContext # type: ignore
from fastapi import HTTPException # type: ignore
import sqlite3
import jwt # type: ignore
from datetime import datetime, timedelta, timezone

# Password hashing context setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "neuralstock_super_secret_key"
ALGORITHM = "HS256"
DB_PATH = "users.db"

def init_db():
    """Create local SQLite table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize DB on script load
init_db()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=24)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def register_user(email, password):
    hashed = pwd_context.hash(password)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (email, hashed_password) VALUES (?, ?)", (email, hashed))
        conn.commit()
        return {"success": True, "msg": "User created successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email is already registered")
    finally:
        conn.close()

def login_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE email=?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if not row or not pwd_context.verify(password, row[0]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    token = create_access_token({"sub": email})
    return {"access_token": token, "token_type": "bearer", "user": email, "success": True}
