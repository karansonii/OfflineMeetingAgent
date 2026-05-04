#core/diarization.py
import requests
import soundfile as sf
import tempfile
import subprocess
import os
import time
from pathlib import Path

from utils.logger import get_logger, log_exception

logger = get_logger("diarization")


class SpeakerDiarization:
    def __init__(self, config):
        self.config = config

        diar_cfg = config.get("diarization", {})

        self.base_url = diar_cfg.get(
            "service_url",
            "http://192.168.7.6:8006"
        )

        self.min_speakers = diar_cfg.get("min_speakers")
        self.max_speakers = diar_cfg.get("max_speakers")
        self.MERGE_THRESHOLD = diar_cfg.get("merge_threshold", 0.40)

        logger.info("\n" + "=" * 60)
        logger.info("  [INIT] Remote Diarization Service")
        logger.info(f"  [INIT] URL: {self.base_url}")
        logger.info("=" * 60)

    def _convert_to_wav(self, audio_path: str) -> tuple[str, bool]:
        if audio_path.lower().endswith(".wav"):
            return audio_path, False

        tmp_wav = tempfile.mktemp(suffix=".wav")

        logger.info(f"  [STEP 1] Converting audio to WAV via ffmpeg...")
        t = time.time()

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                tmp_wav
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            raise RuntimeError(f"ffmpeg conversion failed:\n{err}")

        logger.info(f"  [STEP 1] ✓ Converted in {time.time()-t:.1f}s")
        return tmp_wav, True

    def diarize(self, audio_path):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  [STEP 1] SPEAKER DIARIZATION")
        logger.info(f"  File: {Path(audio_path).name}")
        logger.info(f"{'─'*60}")

        tmp_wav = None

        try:
            wav_path, is_temp = self._convert_to_wav(audio_path)
            tmp_wav = wav_path if is_temp else None

            dur = sf.info(wav_path).duration

            logger.info(f"  [STEP 1] Audio duration : {dur/60:.1f} min ({dur:.0f}s)")
            logger.info(f"  [STEP 1] Calling remote diarization service...")

            t_start = time.time()

            url = self.base_url.rstrip("/") + "/diarize"

            with open(wav_path, "rb") as f:
                files = {
                    "file": (
                        Path(wav_path).name,
                        f,
                        "audio/wav"
                    )
                }

                params = {}

                if self.min_speakers is not None:
                    params["min_speakers"] = self.min_speakers

                if self.max_speakers is not None:
                    params["max_speakers"] = self.max_speakers

                response = requests.post(
                    url,
                    files=files,
                    params=params,
                    timeout=(30, 1800)
                )

            logger.info(f"  [STEP 1] Server response received")
            response.raise_for_status()

            result = response.json()

            elapsed = time.time() - t_start

            logger.info(f"  [STEP 1] ✓ Done in {elapsed:.1f}s")

            segments = []

            for seg in result.get("segments", []):
                segments.append({
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "speaker": seg["speaker"],
                    "duration": float(seg["end"]) - float(seg["start"])
                })

            logger.info(f"  [STEP 1] ✓ Segments: {len(segments)}")

            return segments

        except Exception as e:
            logger.error(f"  [STEP 1] ❌ Error: {e}")
            log_exception(logger, "Diarization failed")
            return self._simple_diarization(audio_path)

        finally:
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass

    def _simple_diarization(self, audio_path):
        logger.info(f"  [STEP 1] Using fallback diarization")

        duration = sf.info(audio_path).duration

        segments = []
        segment_length = 30.0
        current_time = 0.0
        speaker_index = 0

        while current_time < duration:
            end_time = min(current_time + segment_length, duration)

            segments.append({
                "start": current_time,
                "end": end_time,
                "speaker": f"SPEAKER_{speaker_index % 4:02d}",
                "duration": end_time - current_time,
            })

            current_time = end_time
            speaker_index += 1

        return segments