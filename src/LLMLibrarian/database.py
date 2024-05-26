import sqlite3
import json

class Database:
    def __init__(self, db_name):
        self.db_name = db_name
        self.create_database()

    def create_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE,
                chunks TEXT,
                embeddings TEXT,
                summary TEXT,
                summary_embedding TEXT
            )
        """)
        conn.commit()
        conn.close()

    def reset_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS documents")
        conn.commit()
        conn.close()
        self.create_database()

    def insert_document(self, file_path, chunks, embeddings, summary, summary_embedding):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (file_path, chunks, embeddings, summary, summary_embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (file_path, json.dumps(chunks), json.dumps(embeddings), summary, json.dumps(summary_embedding)))
        conn.commit()
        conn.close()

    def get_all_documents(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, chunks, embeddings FROM documents")
        results = cursor.fetchall()
        conn.close()
        return results

    def document_exists(self, file_path):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents WHERE file_path = ?", (file_path,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
