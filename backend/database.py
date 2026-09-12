"""Database Layer for SwasthyaSurge AI (SQLite / PostgreSQL compatible)."""

import sqlite3
import json
import os
from pathlib import Path
from .config import BASE_DIR

DB_PATH = BASE_DIR / "swasthyasurge.db"


def get_db_connection():
    """Establish connection to SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize database tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS districts (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        code TEXT NOT NULL,
        state TEXT NOT NULL,
        population INTEGER,
        base_capacity INTEGER,
        risk_level TEXT,
        surge_probability REAL,
        predicted_demand INTEGER,
        bed_shortage INTEGER,
        metadata_json TEXT
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        severity TEXT NOT NULL,
        district_id INTEGER NOT NULL,
        district_name TEXT NOT NULL,
        reason TEXT NOT NULL,
        predicted_shortage INTEGER,
        surge_probability REAL,
        recommended_action TEXT,
        priority_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS allocations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_month INTEGER NOT NULL,
        district_id INTEGER NOT NULL,
        allocated_units INTEGER NOT NULL,
        expected_shortage REAL,
        surge_risk REAL,
        policy_name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
    conn.close()
