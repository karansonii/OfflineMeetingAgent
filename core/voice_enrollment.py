#core/voice_enrollment.py
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from datetime import datetime
import tempfile
import subprocess
import requests

from utils.logger import get_logger, log_exception

logger = get_logger("voice_enrollment")


class VoiceEnrollment:
    def __init__(self, config):
        self.config = config
        self.min_duration = config['enrollment']['min_duration']

        self.embedding_service_url = config['enrollment'].get(
            'service_url',
            'http://192.168.7.6:8007'
        )

        logger.info("Loading remote speaker embedding service...")
        logger.info(f"✓ Remote embedding URL: {self.embedding_service_url}")

    def enroll_speaker(self, speaker_name, audio_path, role=None):
        try:
            if audio_path.endswith('.m4a') or audio_path.endswith('.mp3'):
                audio_path = self._convert_to_wav(audio_path)

            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr

            if duration < self.min_duration:
                return False, f"Audio too short. Minimum {self.min_duration} seconds required, got {duration:.1f}s"

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()

            sf.write(temp_path, audio, sr)

            embedding = self._extract_embedding(temp_path)

            try:
                Path(temp_path).unlink()
            except Exception as e:
                logger.warning(f"Warning: Could not delete temp file {temp_path}: {e}")

            from database.voice_database import VoiceDatabase
            db = VoiceDatabase(self.config['paths']['speaker_embeddings'])

            speaker_id = db.add_speaker(
                name=speaker_name,
                embedding=embedding,
                role=role,
                enrollment_date=datetime.now().isoformat()
            )

            return True, f"Successfully enrolled {speaker_name}!"

        except Exception as e:
            log_exception(logger, "Enrollment failed")
            return False, f"Enrollment failed: {str(e)}"

    def _convert_to_wav(self, audio_path):
        try:
            import imageio_ffmpeg as ffmpeg

            ffmpeg_path = ffmpeg.get_ffmpeg_exe()
            wav_path = audio_path.rsplit('.', 1)[0] + '.wav'

            command = [
                ffmpeg_path,
                '-i', audio_path,
                '-ar', '16000',
                '-ac', '1',
                '-y',
                wav_path
            ]

            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                raise Exception(f"FFmpeg conversion failed: {result.stderr}")

            return wav_path

        except Exception as e:
            raise

    def _extract_embedding(self, audio_path):
        try:
            url = self.embedding_service_url.rstrip("/") + "/embedding"

            with open(audio_path, "rb") as f:
                files = {
                    "file": (
                        Path(audio_path).name,
                        f,
                        "audio/wav"
                    )
                }

                response = requests.post(
                    url,
                    files=files,
                    timeout=(30, 300)
                )

            response.raise_for_status()

            result = response.json()

            return np.asarray(
                result["embedding"],
                dtype=np.float32
            )

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise