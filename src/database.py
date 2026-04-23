import sqlite3
import os
import yaml
from datetime import datetime


class ALPRDatabase:
    def __init__(self, config_path=None):
        if config_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base, "config.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        db_path = cfg["database"]["db_path"]

        # Make path absolute relative to project root
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_path = os.path.join(base, db_path)

        self._init_db()

    # ------------------------------------------------------------------
    # Initialize database and create table
    # ------------------------------------------------------------------
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plate_reads (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id     INTEGER NOT NULL,
                    plate_text   TEXT    NOT NULL,
                    confidence   REAL    NOT NULL,
                    timestamp    TEXT    NOT NULL,
                    frame_number INTEGER NOT NULL
                )
            """)
            conn.commit()
        print(f"[Database] Initialized at: {self.db_path}")

    # ------------------------------------------------------------------
    # Insert a new plate read
    # ------------------------------------------------------------------
    def insert_read(self, track_id, plate_text, confidence, frame_number):
        # Prevent duplicate: skip if same track_id has confidence >= 80
        existing = self._get_best_read_for_track(track_id)
        if existing and existing["confidence"] >= 80.0:
            print(f"[Database] Skipping duplicate for track_id={track_id}")
            return False

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO plate_reads
                    (track_id, plate_text, confidence, timestamp, frame_number)
                VALUES (?, ?, ?, ?, ?)
            """, (track_id, plate_text, confidence, timestamp, frame_number))
            conn.commit()
        print(f"[Database] Inserted: {plate_text} (track={track_id}, conf={confidence:.1f})")
        return True

    # ------------------------------------------------------------------
    # Get best existing read for a track_id
    # ------------------------------------------------------------------
    def _get_best_read_for_track(self, track_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT * FROM plate_reads
                WHERE track_id = ?
                ORDER BY confidence DESC
                LIMIT 1
            """, (track_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------
    # Get all reads
    # ------------------------------------------------------------------
    def get_all_reads(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT * FROM plate_reads
                ORDER BY id DESC
            """)
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Get reads filtered by date (YYYY-MM-DD)
    # ------------------------------------------------------------------
    def get_reads_by_date(self, date_str):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT * FROM plate_reads
                WHERE timestamp LIKE ?
                ORDER BY id DESC
            """, (f"{date_str}%",))
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Get unique plate texts
    # ------------------------------------------------------------------
    def get_unique_plates(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("""
                SELECT DISTINCT plate_text, MAX(confidence) as confidence,
                       MAX(timestamp) as timestamp
                FROM plate_reads
                GROUP BY plate_text
                ORDER BY timestamp DESC
            """)
            return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Clear all records
    # ------------------------------------------------------------------
    def clear_all(self):
     with sqlite3.connect(self.db_path) as conn:
        conn.execute("DELETE FROM plate_reads")
        # Reset auto-increment counter
        conn.execute("DELETE FROM sqlite_sequence WHERE name='plate_reads'")
        conn.commit()
    print("[Database] All records cleared and ID reset.")