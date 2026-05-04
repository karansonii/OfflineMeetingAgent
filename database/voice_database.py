#database/voice_database.py
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import uuid

from utils.logger import get_logger

logger = get_logger("voice_database")


class VoiceDatabase:
    def __init__(self, storage_path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.db_file = self.storage_path / "speakers.json"
        self.embeddings_dir = self.storage_path / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)

        self._load_database()

    def _load_database(self):
        """Load speaker database"""
        if self.db_file.exists():
            with open(self.db_file, 'r') as f:
                self.speakers = json.load(f)
        else:
            self.speakers = {}
            self._save_database()

    def _save_database(self):
        """Save speaker database"""
        with open(self.db_file, 'w') as f:
            json.dump(self.speakers, f, indent=2)

    def add_speaker(self, name, embedding, role=None, enrollment_date=None):
        """Add new speaker"""
        # Clean name
        clean_name = name.strip().lower().replace(" ", "_")

        speaker_id = clean_name

        # Save embedding
        embedding_path = self.embeddings_dir / f"{clean_name}.npy"
        np.save(embedding_path, embedding)

        # Add to database
        self.speakers[speaker_id] = {
            'id': speaker_id,
            'name': name,
            'role': role,
            'enrollment_date': enrollment_date or datetime.now().isoformat(),
            'embedding_path': str(embedding_path)
        }

        self._save_database()
        return speaker_id

    def get_speaker(self, speaker_id):
        """Get speaker info"""
        if speaker_id not in self.speakers:
            return None

        speaker = self.speakers[speaker_id].copy()

        # Load embedding
        embedding_path = Path(speaker['embedding_path'])
        if embedding_path.exists():
            speaker['embedding'] = np.load(embedding_path)

        return speaker

    def get_all_speakers(self):
        """Get all speakers with embeddings"""
        speakers = []
        for speaker_id in self.speakers:
            speaker = self.get_speaker(speaker_id)
            if speaker:
                speakers.append(speaker)
        return speakers

    def delete_speaker(self, speaker_id):
        """Delete speaker"""
        if speaker_id not in self.speakers:
            return False

        # Delete embedding file
        embedding_path = Path(self.speakers[speaker_id]['embedding_path'])
        if embedding_path.exists():
            embedding_path.unlink()

        # Remove from database
        del self.speakers[speaker_id]
        self._save_database()

        return True