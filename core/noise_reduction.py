#core/noise_reduction.py
import time
import subprocess
import tempfile
import os
from pathlib import Path

from utils.logger import get_logger, log_exception

logger = get_logger("noise_reduction")


class NoiseReducer:
    def __init__(self):
        logger.info("\n" + "="*60)
        logger.info("  [INIT] Loading DeepFilterNet model...")
        t = time.time()
        from df.enhance import enhance, init_df, load_audio, save_audio
        self._enhance    = enhance
        self._load_audio = load_audio
        self._save_audio = save_audio
        self.model, self.df_state, _ = init_df()
        logger.info(f"  [INIT] ✓ DeepFilterNet loaded in {time.time()-t:.1f}s")
        logger.info("="*60)

    # ─────────────────────────────────────────────────────────────────────

    def _convert_to_wav(self, input_path: str) -> tuple[str, bool]:
        """
        Convert any audio to a clean WAV using ffmpeg so DeepFilterNet
        never has to touch a potentially corrupt MP3 header.
        Returns (wav_path, is_temp).  Caller must delete if is_temp=True.
        """
        if input_path.lower().endswith(".wav"):
            return input_path, False

        tmp_wav = tempfile.mktemp(suffix=".wav")
        logger.info(f"  [STEP 0] Converting audio to WAV via ffmpeg...")
        t = time.time()
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ar", str(self.df_state.sr()),  # match DeepFilterNet's expected SR
                "-ac", "1",                       # mono
                "-f", "wav",
                tmp_wav,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            err = result.stderr.decode(errors="replace")
            raise RuntimeError(f"ffmpeg conversion failed:\n{err}")
        logger.info(f"  [STEP 0] ✓ Converted to WAV in {time.time()-t:.1f}s")
        return tmp_wav, True

    # ─────────────────────────────────────────────────────────────────────

    def enhance_audio(self, input_path: str) -> str:
        input_path  = Path(input_path)
        output_path = input_path.parent / f"{input_path.stem}_enhanced.wav"

        logger.info(f"\n{'─'*60}")
        logger.info(f"  [STEP 0] NOISE REDUCTION")
        logger.info(f"  Input  : {input_path.name}")
        logger.info(f"  Output : {output_path.name}")
        logger.info(f"{'─'*60}")

        t_start = time.time()
        tmp_wav = None
        try:
            # ── Convert to clean WAV first ────────────────────────────
            wav_path, is_temp = self._convert_to_wav(str(input_path))
            tmp_wav = wav_path if is_temp else None

            # ── Load ──────────────────────────────────────────────────
            logger.info(f"  [STEP 0] Loading audio...")
            t1 = time.time()
            audio, _ = self._load_audio(wav_path, sr=self.df_state.sr())
            logger.info(f"  [STEP 0] Audio loaded in {time.time()-t1:.2f}s")

            # ── Enhance ───────────────────────────────────────────────
            logger.info(f"  [STEP 0] Running DeepFilterNet enhancement...")
            t2 = time.time()
            enhanced = self._enhance(self.model, self.df_state, audio)
            logger.info(f"  [STEP 0] Enhancement done in {time.time()-t2:.2f}s")

            # ── Save ──────────────────────────────────────────────────
            logger.info(f"  [STEP 0] Saving enhanced audio...")
            t3 = time.time()
            self._save_audio(str(output_path), enhanced, self.df_state.sr())
            logger.info(f"  [STEP 0] Saved in {time.time()-t3:.2f}s")

            elapsed = time.time() - t_start
            logger.info(f"  [STEP 0] ✓ NOISE REDUCTION COMPLETE — total: {elapsed:.1f}s")
            logger.info(f"{'─'*60}\n")
            return str(output_path)

        except Exception as e:
            elapsed = time.time() - t_start
            logger.warning(f"  [STEP 0] ⚠️  DeepFilterNet failed after {elapsed:.1f}s: {e}")
            logger.warning(f"  [STEP 0] → Falling back to original audio")
            logger.info(f"{'─'*60}\n")
            return str(input_path)

        finally:
            # ── Clean up temp WAV ─────────────────────────────────────
            if tmp_wav and os.path.exists(tmp_wav):
                try:
                    os.remove(tmp_wav)
                except Exception:
                    pass