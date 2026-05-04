#core/transcription.py
import re
import requests
import time
import soundfile as sf
from pathlib import Path

from utils.logger import get_logger, log_exception

logger = get_logger("transcription")


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination / repetition detection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_repeated_phrases(text: str, min_phrase_words: int = 3, max_repeats: int = 2) -> str:
    """
    Detect and collapse repeating phrase loops that Whisper hallucinates.

    Example input : "I'm a little bit of a. I'm a little bit of a. I'm a little bit of a."
    Example output: "I'm a little bit of a."

    Strategy:
      1. Split text into sentences.
      2. For each sentence, check if it (or a very similar version) appears
         more than max_repeats times consecutively. If so, keep only one copy.
      3. Also check for sub-sentence word-level repetition loops.
    """
    if not text or not text.strip():
        return text

    # ── Pass 1: sentence-level dedup ─────────────────────────────────────
    # Split on sentence boundaries (.  !  ?)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    deduped = []
    i = 0
    while i < len(sentences):
        sent = sentences[i].strip()
        if not sent:
            i += 1
            continue
        # Count how many times this exact (case-insensitive) sentence repeats from here
        count = 1
        j = i + 1
        while j < len(sentences) and sentences[j].strip().lower() == sent.lower():
            count += 1
            j += 1
        if count > max_repeats:
            logger.info(f"    [dedup] Sentence repeated {count}× — keeping 1: '{sent[:60]}'")
        deduped.append(sent)
        i = j  # skip all the duplicates

    text = ' '.join(deduped)

    # ── Pass 2: word-ngram loop detection ─────────────────────────────────
    # Detect patterns like "word1 word2 word3 word1 word2 word3 word1 word2 word3"
    words = text.split()
    if len(words) < min_phrase_words * 2:
        return text

    result_words = []
    i = 0
    while i < len(words):
        # Try phrase lengths from large to small to find the longest repeating unit
        found_loop = False
        for phrase_len in range(min(20, len(words) - i), min_phrase_words - 1, -1):
            phrase = words[i:i + phrase_len]
            # Count consecutive repetitions of this phrase starting at i
            reps = 1
            j = i + phrase_len
            while j + phrase_len <= len(words) and words[j:j + phrase_len] == phrase:
                reps += 1
                j += phrase_len
            if reps > max_repeats:
                phrase_str = ' '.join(phrase)
                logger.info(f"    [dedup] Phrase repeated {reps}× — keeping 1: '{phrase_str[:60]}'")
                result_words.extend(phrase)  # keep exactly one copy
                i = j  # skip all repetitions
                found_loop = True
                break
        if not found_loop:
            result_words.append(words[i])
            i += 1

    return ' '.join(result_words)


def _is_hallucinated_segment(text: str) -> bool:
    """
    Return True if a segment looks like a Whisper hallucination that should be dropped.

    Common Whisper hallucinations on non-English/low-quality audio:
    - Very short filler transcriptions repeated endlessly
    - Segments that are pure music/sound descriptions: [MUSIC], (applause) etc.
    - Segments with abnormally high ratio of punctuation or special chars
    - Segments that are just "Thank you.", "Thanks.", etc. repeated (no real content)
    """
    if not text or not text.strip():
        return True

    stripped = text.strip()

    # Drop segments that are just bracketed sound descriptions
    if re.match(r'^\[.*\]$', stripped) or re.match(r'^\(.*\)$', stripped):
        logger.info(f"    [hallucination] Dropped sound-description segment: '{stripped}'")
        return True

    # Drop segments that are ONLY punctuation / whitespace
    if not re.search(r'[a-zA-Z\u0900-\u097F\u0A80-\u0AFF]', stripped):
        logger.info(f"    [hallucination] Dropped no-letter segment: '{stripped}'")
        return True

    # Drop known Whisper hallucination phrases
    HALLUCINATION_PHRASES = {
        'thank you', 'thanks', 'thank you for watching', 'thanks for watching',
        'please subscribe', 'like and subscribe', 'see you next time',
        'bye bye', 'goodbye', 'good bye',
        'subtitles by', 'subtitles were', 'transcribed by',
        'you', '.', '...', '…',
    }
    if stripped.lower().rstrip('.!?,') in HALLUCINATION_PHRASES:
        logger.info(f"    [hallucination] Dropped known hallucination: '{stripped}'")
        return True

    # Drop if >60% of characters are non-ASCII / punctuation (garbled multilingual)
    total   = len(stripped)
    letters = len(re.findall(r'[a-zA-Z0-9\u0900-\u097F\u0A80-\u0AFF\s]', stripped))
    if total > 10 and (letters / total) < 0.5:
        logger.info(f"    [hallucination] Dropped high-noise segment: '{stripped[:60]}'")
        return True

    return False


def _clean_segment_text(text: str) -> str:
    """
    Apply all text cleaning steps to a single segment's text:
    1. Strip leading/trailing whitespace
    2. Remove repeated phrases/sentences (Whisper hallucination loops)
    3. Collapse multiple spaces
    """
    if not text:
        return text

    text = text.strip()

    # Remove Whisper-style repetition loops
    text = _remove_repeated_phrases(text)

    # Collapse multiple spaces / newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main transcriber
# ─────────────────────────────────────────────────────────────────────────────

class WhisperTranscriber:
    def __init__(self, config):
        self.config = config

        whisper_cfg = config.get("whisper", {})

        self.base_url = whisper_cfg.get(
            "service_url",
            "http://192.168.7.6:8000"
        )

        self.language = whisper_cfg.get("language", None)

        logger.info(f"\n{'='*60}")
        logger.info(f"  [INIT] Remote Faster-Whisper Service")
        logger.info(f"  [INIT] URL: {self.base_url}")
        logger.info(f"{'='*60}")

    def transcribe(self, audio_path):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  [STEP 2] TRANSCRIPTION & TRANSLATION")
        logger.info(f"  File: {Path(audio_path).name}")
        logger.info(f"{'─'*60}")

        dur = sf.info(audio_path).duration

        logger.info(f"  [STEP 2] Audio duration : {dur/60:.1f} min ({dur:.0f}s)")
        logger.info(f"  [STEP 2] Calling remote Whisper service...")

        t_start = time.time()

        url = self.base_url.rstrip("/") + "/transcribe"

        with open(audio_path, "rb") as f:
            files = {
                "file": (
                    Path(audio_path).name,
                    f,
                    "audio/wav"
                )
            }

            params = {}
            if self.language:
                params["language"] = self.language

            response = requests.post(
                url,
                files=files,
                params=params,
                timeout=(30, 1200)
            )

        response.raise_for_status()

        result = response.json()

        elapsed = time.time() - t_start
        rtf = elapsed / dur if dur > 0 else 0

        logger.info(f"  [STEP 2] ✓ Done in {elapsed:.1f}s")
        logger.info(f"  [STEP 2] RTF : {rtf:.2f}x")

        segments = []
        dropped  = 0

        # ── CASE 1: Server returns proper segments ────────────────────
        if "segments" in result and result["segments"]:
            for seg in result["segments"]:
                raw_text = seg.get("text", "").strip()

                # Clean repeated phrases first
                cleaned_text = _clean_segment_text(raw_text)

                # Drop hallucinated segments
                if _is_hallucinated_segment(cleaned_text):
                    dropped += 1
                    logger.info(f"    [STEP 2] Dropped hallucinated segment: '{raw_text[:80]}'")
                    continue

                # Also clean word-level timestamps if present
                words = seg.get("words", [])
                cleaned_words = _clean_words(words)

                segments.append({
                    "start": float(seg.get("start", 0)),
                    "end":   float(seg.get("end", 0)),
                    "text":  cleaned_text,
                    "words": cleaned_words,
                })

        # ── CASE 2: Server returns plain text only ────────────────────
        elif "text" in result:
            raw_text     = result["text"].strip()
            cleaned_text = _clean_segment_text(raw_text)
            if not _is_hallucinated_segment(cleaned_text):
                segments.append({
                    "start": 0.0,
                    "end":   float(dur),
                    "text":  cleaned_text,
                    "words": [],
                })
            else:
                dropped += 1

        logger.info(f"  [STEP 2] Raw segments   : {len(segments) + dropped}")
        logger.info(f"  [STEP 2] After cleaning : {len(segments)} (dropped {dropped} hallucinated)")

        return segments


# ─────────────────────────────────────────────────────────────────────────────
# Word-level repetition cleaner
# ─────────────────────────────────────────────────────────────────────────────

def _clean_words(words: list) -> list:
    """
    Remove consecutive duplicate words from the word-timestamp list.
    Whisper sometimes emits the same word twice with slightly different timestamps.
    """
    if not words:
        return words

    cleaned = []
    prev_word = None

    for w in words:
        word_text = w.get('word', '').strip().lower()
        if word_text and word_text != prev_word:
            cleaned.append(w)
        prev_word = word_text

    return cleaned