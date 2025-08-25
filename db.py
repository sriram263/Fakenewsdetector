# db.py
import psycopg2
from dotenv import load_dotenv
import os

# Load credentials
load_dotenv()

def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def insert_chat(speaker, message, prediction="Null", confidence_score=0.00):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO chat_test (speaker, message, prediction, confidence_score)
            VALUES (%s, %s, %s, %s);
        """, (speaker, message, prediction, confidence_score))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("❌ DB Insert Error:", e)
