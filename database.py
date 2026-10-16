```python
import sqlite3
import json
from pathlib import Path

DATABASE_PATH = Path("face_tracker.db")


def create_database():
    """Create the database and its tables if they do not exist."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                embedding TEXT,
                UNIQUE(track_id, image_path)
            )
        """)

        # Add the embedding column to an existing database if needed.
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(faces)"
            ).fetchall()
        }

        if "embedding" not in columns:
            connection.execute(
                "ALTER TABLE faces ADD COLUMN embedding TEXT"
            )

        connection.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT
            )
        """)

    print(f"Database ready: {DATABASE_PATH.resolve()}")


def add_face(track_id, image_path, timestamp, embedding=None):
    """Save a face crop and its optional embedding."""
    embedding_text = (
        json.dumps(embedding.tolist())
        if embedding is not None
        else None
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT OR IGNORE INTO faces
                (track_id, image_path, first_seen, embedding)
            VALUES (?, ?, ?, ?)
        """, (track_id, image_path, timestamp, embedding_text))


def add_event(track_id, event_type, timestamp, image_path=None):
    """Save an event associated with a tracked face."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            INSERT INTO events
                (track_id, event_type, timestamp, image_path)
            VALUES (?, ?, ?, ?)
        """, (track_id, event_type, timestamp, image_path))


if __name__ == "__main__":
    create_database()
```