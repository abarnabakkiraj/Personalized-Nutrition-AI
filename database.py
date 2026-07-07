"""
database.py - Database Management Module
Handles all SQLite database operations for the Personalized Nutrition AI system.
"""

import sqlite3
import bcrypt
import json
from datetime import datetime


import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "nutrition_ai.db")

def get_connection():
    print("Database file:", DB_PATH)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database and creates all required tables if they don't exist.
    Tables: Users, BloodReports, Recommendations, PredictionHistory
    """
    conn = get_connection()
    cursor = conn.cursor()

    # ── Table 1: Users ──────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            full_name   TEXT,
            age         INTEGER,
            gender      TEXT,
            email       TEXT,
            created_at  TEXT    DEFAULT (datetime('now'))
        )
    """)

    # ── Table 2: BloodReports ───────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS BloodReports (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            report_date     TEXT    DEFAULT (datetime('now')),
            hemoglobin      REAL,
            glucose         REAL,
            cholesterol     REAL,
            vitamin_d       REAL,
            iron            REAL,
            calcium         REAL,
            vitamin_b12     REAL,
            triglycerides   REAL,
            source_type     TEXT,   -- 'manual', 'csv', 'ocr'
            FOREIGN KEY (user_id) REFERENCES Users(id)
        )
    """)

    # ── Table 3: Recommendations ────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Recommendations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id       INTEGER NOT NULL,
            user_id         INTEGER NOT NULL,
            parameter       TEXT,
            status          TEXT,   -- 'Low', 'Normal', 'High'
            recommendation  TEXT,   -- JSON list of food suggestions
            created_at      TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (report_id) REFERENCES BloodReports(id),
            FOREIGN KEY (user_id)   REFERENCES Users(id)
        )
    """)

    # ── Table 4: PredictionHistory ──────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS PredictionHistory (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            report_id       INTEGER,
            risk_level      TEXT,   -- 'Low', 'Medium', 'High'
            health_score    REAL,
            predicted_at    TEXT    DEFAULT (datetime('now')),
            model_version   TEXT    DEFAULT '1.0',
            FOREIGN KEY (user_id)  REFERENCES Users(id),
            FOREIGN KEY (report_id) REFERENCES BloodReports(id)
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")


# ── User Auth Functions ──────────────────────────────────────────────────────

def register_user(username, password, full_name, age, gender, email):
    """Registers a new user with hashed password. Returns (True, msg) or (False, error)."""
    try:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Users (username, password, full_name, age, gender, email)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (username, hashed, full_name, age, gender, email))
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."
    except Exception as e:
        return False, str(e)


def login_user(username, password):
    """
    Validates login credentials.
    Returns user dict on success, None on failure.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if user and bcrypt.checkpw(password.encode(), user["password"].encode()):
        return dict(user)
    return None


# ── Blood Report Functions ───────────────────────────────────────────────────

def save_blood_report(user_id, values: dict, source_type="manual"):
    """
    Saves extracted blood parameter values to BloodReports table.
    Returns the new report_id.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO BloodReports
            (user_id, hemoglobin, glucose, cholesterol, vitamin_d,
             iron, calcium, vitamin_b12, triglycerides, source_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        values.get("hemoglobin"),
        values.get("glucose"),
        values.get("cholesterol"),
        values.get("vitamin_d"),
        values.get("iron"),
        values.get("calcium"),
        values.get("vitamin_b12"),
        values.get("triglycerides"),
        source_type
    ))
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id


def get_user_reports(user_id):
    """Returns all blood reports for a given user, newest first."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM BloodReports WHERE user_id = ?
        ORDER BY report_date DESC
    """, (user_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


# ── Recommendation Functions ─────────────────────────────────────────────────

def save_recommendations(user_id, report_id, recs: list):
    """
    Saves a list of recommendation dicts to the Recommendations table.
    Each rec: {'parameter': str, 'status': str, 'recommendation': [foods]}
    """
    conn = get_connection()
    cursor = conn.cursor()
    for rec in recs:
        cursor.execute("""
            INSERT INTO Recommendations (report_id, user_id, parameter, status, recommendation)
            VALUES (?, ?, ?, ?, ?)
        """, (
            report_id,
            user_id,
            rec["parameter"],
            rec["status"],
            json.dumps(rec["recommendation"])
        ))
    conn.commit()
    conn.close()


def get_recommendations(report_id):
    """Returns all recommendations for a given report."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Recommendations WHERE report_id = ?", (report_id,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["recommendation"] = json.loads(d["recommendation"])
        result.append(d)
    return result


# ── Prediction History Functions ─────────────────────────────────────────────

def save_prediction(user_id, report_id, risk_level, health_score):
    """Saves ML prediction results to PredictionHistory."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO PredictionHistory (user_id, report_id, risk_level, health_score)
        VALUES (?, ?, ?, ?)
    """, (user_id, report_id, risk_level, health_score))
    conn.commit()
    conn.close()


def get_prediction_history(user_id):
    """Returns all past predictions for a user."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ph.*, br.report_date
        FROM PredictionHistory ph
        LEFT JOIN BloodReports br ON ph.report_id = br.id
        WHERE ph.user_id = ?
        ORDER BY ph.predicted_at DESC
    """, (user_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows
