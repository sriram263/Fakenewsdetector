import psycopg2
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

try:
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    print("✅ Connected to the database successfully!")

    cur = conn.cursor()

    # Drop old table if it exists
    cur.execute("DROP TABLE IF EXISTS chat_test;")

    # Create new table with default values
    cur.execute("""
        CREATE TABLE chat_test (
            id SERIAL PRIMARY KEY,
            time_record TIMESTAMP(0) DEFAULT CURRENT_TIMESTAMP,
            speaker VARCHAR(10),
            message TEXT,
            prediction VARCHAR(10) DEFAULT 'UNKNOWN',
            confidence_score NUMERIC(5,2) DEFAULT 0.00
        );
    """)
    conn.commit()
    print("✅ Created new table 'chat_test' with default values.")

    # Sample insert without prediction/confidence (defaults will be used)
    sample_data = [
        ("user", "Hello, how are you?"),
        ("bot", "I am a chatbot, ready to assist you!")
    ]
    cur.executemany("""
        INSERT INTO chat_test (speaker, message)
        VALUES (%s, %s);
    """, sample_data)
    conn.commit()
    print("✅ Inserted sample data using default values.")

    # Close connection
    cur.close()
    conn.close()
    print("\n✅ Connection closed.")

except Exception as e:
    print("❌ Error:", e)
