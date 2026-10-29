"""Local patient authentication and history storage for the Streamlit app.

Uses SQLite and PBKDF2 password hashing from Python's standard library.
This is suitable for a local academic demo; production deployments need
HTTPS, secrets management, rate limiting, and a security review.
"""
import base64
import hashlib
import hmac
import os
import secrets
import sqlite3

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(PROJECT_DIR, "data", "patients.db")
_ITERATIONS = 310_000


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                hashed_password TEXT NOT NULL,
                date_of_birth TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patient_profiles (
                user_id INTEGER PRIMARY KEY,
                gender TEXT DEFAULT '',
                existing_conditions TEXT DEFAULT '',
                current_medications TEXT DEFAULT '',
                allergies TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                symptoms TEXT NOT NULL,
                primary_condition TEXT,
                confidence REAL,
                top_3_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_b64, digest_b64 = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, int(iterations)
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def register_patient(full_name: str, email: str, password: str, date_of_birth: str):
    init_db()
    full_name, email = full_name.strip(), email.strip().lower()
    if not full_name or not email or not date_of_birth:
        raise ValueError("Please complete all required fields.")
    if len(password) < 10:
        raise ValueError("Use a password with at least 10 characters.")
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        raise ValueError("Enter a valid email address.")
    hashed = _hash_password(password)
    try:
        with _connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users (full_name, email, hashed_password, date_of_birth) VALUES (?, ?, ?, ?)",
                (full_name, email, hashed, date_of_birth),
            )
            user_id = cursor.lastrowid
            conn.execute("INSERT INTO patient_profiles (user_id) VALUES (?)", (user_id,))
            return user_id
    except sqlite3.IntegrityError as exc:
        if "email" in str(exc).lower():
            raise ValueError("An account with that email already exists. Please sign in.") from exc
        raise


def authenticate_patient(email: str, password: str):
    init_db()
    with _connect() as conn:
        user = conn.execute(
            "SELECT id, full_name, email, date_of_birth, created_at, hashed_password "
            "FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip(),),
        ).fetchone()
    if user and _verify_password(password, user["hashed_password"]):
        result = dict(user)
        result.pop("hashed_password", None)
        return result
    return None


def save_prediction(user_id: int, symptoms: list, primary_condition: str,
                    confidence: float, top_3_json: str):
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO prediction_history "
            "(user_id, symptoms, primary_condition, confidence, top_3_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, ", ".join(symptoms), primary_condition, confidence, top_3_json),
        )


def get_patient_history(user_id: int):
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, symptoms, primary_condition, confidence, top_3_json, created_at "
            "FROM prediction_history WHERE user_id = ? ORDER BY created_at DESC, id DESC LIMIT 50",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_patient_profile(user_id: int):
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT gender, existing_conditions, current_medications, allergies "
            "FROM patient_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else {
        "gender": "", "existing_conditions": "", "current_medications": "", "allergies": ""
    }


def update_patient_profile(user_id: int, gender: str, existing_conditions: str,
                           current_medications: str, allergies: str):
    init_db()
    with _connect() as conn:
        conn.execute("""
            INSERT INTO patient_profiles
                (user_id, gender, existing_conditions, current_medications, allergies, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                gender=excluded.gender,
                existing_conditions=excluded.existing_conditions,
                current_medications=excluded.current_medications,
                allergies=excluded.allergies,
                updated_at=CURRENT_TIMESTAMP
        """, (user_id, gender, existing_conditions, current_medications, allergies))
