#core/speaker_identifier.py
import torch
import librosa
import numpy as np
import re
from pathlib import Path
from pyannote.audio import Model, Inference
import requests
import tempfile
import soundfile as sf
import time

from utils.logger import get_logger

logger = get_logger("speaker_identifier")


def _remove_repeated_phrases_simple(text: str, min_len: int = 3, max_reps: int = 2) -> str:
    """
    Collapse word-level repetition loops in a text string.
    Example: "I'm a little bit of a I'm a little bit of a I'm a little bit of a"
          →  "I'm a little bit of a"
    """
    words = text.split()
    if len(words) < min_len * 2:
        return text

    result = []
    i = 0
    while i < len(words):
        found = False
        for phrase_len in range(min(15, len(words) - i), min_len - 1, -1):
            phrase = words[i:i + phrase_len]
            reps = 1
            j = i + phrase_len
            while j + phrase_len <= len(words) and words[j:j + phrase_len] == phrase:
                reps += 1
                j += phrase_len
            if reps > max_reps:
                result.extend(phrase)
                i = j
                found = True
                break
        if not found:
            result.append(words[i])
            i += 1
    return ' '.join(result)


def _truncate_at_word_boundary(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(' ')
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated.rstrip()


class SpeakerIdentifier:
    def __init__(self, config, voice_db):
        self.config   = config
        self.voice_db = voice_db

        logger.info(f"\n{'='*60}")
        logger.info("  [INIT] Remote SpeakerIdentifier embedding service")
        t = time.time()

        self.embedding_service_url = config.get('enrollment', {}).get(
            'service_url',
            'http://192.168.7.6:8007'
        )

        logger.info(f"  [INIT] URL: {self.embedding_service_url}")
        logger.info(f"  [INIT] ✓ Ready in {time.time()-t:.1f}s")
        logger.info(f"{'='*60}")

    # ─────────────────────────────────────────────────────────────────────

    def identify_speakers(self, audio_path, speaker_segments, transcription):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  [STEP 3] SPEAKER IDENTIFICATION & ALIGNMENT")
        logger.info(f"  File            : {Path(audio_path).name}")
        logger.info(f"  Diar. segments  : {len(speaker_segments)}")
        logger.info(f"  Whisper segments: {len(transcription)}")
        logger.info(f"{'─'*60}")
        t_total = time.time()

        unique_labels = sorted(set(s['speaker'] for s in speaker_segments))
        logger.info(f"  [STEP 3] Unique speaker labels from diarization: {unique_labels}")

        audio, sr = librosa.load(audio_path, sr=16000)
        enrolled  = self.voice_db.get_all_speakers()

        # ── 3a. Match against enrolled speakers ──────────────────────
        logger.info(f"\n  [STEP 3a] Speaker matching against enrolled voices...")
        t3a = time.time()
        if enrolled:
            spk_embeddings = {s['id']: s['embedding'] for s in enrolled}
            spk_names      = {s['id']: s['name']      for s in enrolled}
            logger.info(f"  [STEP 3a] Enrolled speakers: {len(enrolled)}")
            for s in enrolled:
                logger.info(f"    [enrolled] id={s['id']}  name={s['name']}")
            raw_ids = self._identify_all_segments(audio, sr, speaker_segments, spk_embeddings, spk_names)
        else:
            logger.info(f"  [STEP 3a] No enrolled speakers — using generic labels")
            raw_ids = {}
        logger.info(f"  [STEP 3a] ✓ Matching done in {time.time()-t3a:.2f}s")
        logger.info(f"  [STEP 3a] raw_ids result: {raw_ids}")

        # ── 3b. Build display name map ────────────────────────────────
        logger.info(f"\n  [STEP 3b] Building display name map...")
        t3b         = time.time()
        display_map = self._build_display_name_map(speaker_segments, raw_ids)
        logger.info(f"  [STEP 3b] ✓ Display map built in {time.time()-t3b:.2f}s")
        logger.info(f"  [STEP 3b] Final name map: {display_map}")

        # ── 3c. Align transcription to speakers ───────────────────────
        logger.info(f"\n  [STEP 3c] Aligning Whisper words to speaker segments...")
        t3c    = time.time()
        result = self._align_transcription(speaker_segments, transcription, display_map)
        logger.info(f"  [STEP 3c] ✓ Alignment done in {time.time()-t3c:.2f}s  →  {len(result)} aligned segments")

        # ── 3d. Merge consecutive SAME-SPEAKER segments only ─────────
        logger.info(f"\n  [STEP 3d] Merging consecutive same-speaker segments...")
        t3d    = time.time()
        result = merge_consecutive_speaker_segments(result, max_gap=5.0)  # FIX: increased from 3.0 to 5.0
        logger.info(f"  [STEP 3d] ✓ Merge done in {time.time()-t3d:.2f}s  →  {len(result)} final segments")

        # ── 3e. Debug: print final speaker breakdown ──────────────────
        logger.info(f"\n  [STEP 3e] Final segment breakdown:")
        for seg in result:
            logger.info(f"    [{seg['start']:.1f}s-{seg['end']:.1f}s] {seg['speaker_name']}: {seg['text'][:60]}...")

        logger.info(f"\n  [STEP 3] ✓ IDENTIFICATION COMPLETE — total: {time.time()-t_total:.1f}s")
        logger.info(f"{'─'*60}\n")
        return result

    # ─────────────────────────────────────────────────────────────────────

    def _identify_all_segments(self, audio, sr, speaker_segments, spk_embeddings, spk_names):
        """
        FIX FOR ISSUES 1 & 2:

        Issue 1 — Same person, two diarization labels (e.g. SPEAKER_00 and SPEAKER_01):
          After individual matching, we do a cross-label similarity check.
          If two unmatched labels are highly similar to each other (>0.75 cosine),
          and one of them matched an enrolled speaker, the other gets the same name.
          This handles diarization over-segmentation of a single speaker.

        Issue 2 — Unknown speaker falsely matched to enrolled speaker:
          Raised effective threshold logic: we now require confidence >= match_threshold
          AND the match must beat the second-best enrolled speaker by a margin (0.08).
          This prevents borderline matches from claiming an enrolled identity.
        """
        # Group all segments by label
        label_to_segs = {}
        for seg in speaker_segments:
            label = seg['speaker']
            if label not in label_to_segs:
                label_to_segs[label] = []
            label_to_segs[label].append(seg)

        identifications = {}
        threshold = self.config.get('enrollment', {}).get('match_threshold', 0.55)
        # FIX Issue 2: require the top match to beat second-best by this margin
        MIN_MARGIN = 0.08

        logger.info(f"    [match] Using match_threshold={threshold}, min_margin={MIN_MARGIN}")
        logger.info(f"    [match] Embedding {len(label_to_segs)} unique speaker labels...")

        # Store all label embeddings for cross-label dedup (Fix Issue 1)
        label_embeddings = {}

        for label, segs in label_to_segs.items():
            t_emb = time.time()
            logger.info(f"    [match] {label}: {len(segs)} segment(s), trying multi-window embedding...")

            emb = self._embed_label(audio, sr, segs)
            if emb is None:
                logger.info(f"    [match] {label} — could not extract embedding, skipping")
                continue

            label_embeddings[label] = emb  # store for cross-label check later

            # Compute similarity against all enrolled speakers
            scores = {}
            for sid, enrolled_emb in spk_embeddings.items():
                ef   = np.array(enrolled_emb).flatten()
                tf   = emb.flatten()
                norm = np.linalg.norm(tf) * np.linalg.norm(ef)
                if norm == 0:
                    continue
                sim = float(np.dot(tf, ef) / norm)
                scores[sid] = sim
                logger.info(f"      [match] {label} vs {spk_names[sid]:20s}: sim={sim:.3f}")

            if not scores:
                continue

            # Sort by score descending
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            best_id, best_sim = sorted_scores[0]
            second_sim = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
            margin = best_sim - second_sim

            elapsed_emb = time.time() - t_emb

            # FIX Issue 2: require BOTH threshold AND margin to avoid false matches
            if best_sim >= threshold and margin >= MIN_MARGIN:
                identifications[label] = {
                    'id':         best_id,
                    'name':       spk_names[best_id],
                    'confidence': best_sim,
                }
                logger.info(f"    [match] ✅ {label} → {spk_names[best_id]:20s} "
                            f"(conf={best_sim:.3f}, margin={margin:.3f}, {elapsed_emb:.2f}s)")
            elif best_sim >= threshold and margin < MIN_MARGIN:
                logger.info(f"    [match] ❌ {label} → REJECTED despite sim={best_sim:.3f} "
                            f"(margin={margin:.3f} < {MIN_MARGIN} — ambiguous match, treating as unknown)")
            else:
                logger.info(f"    [match] ❌ {label} → no match "
                            f"(best={spk_names[best_id]}, sim={best_sim:.3f} < {threshold})")

        # ── FIX Issue 1: Cross-label deduplication ────────────────────────
        # If an unmatched label is very similar to a matched label's embedding,
        # assign it the same enrolled name (same physical speaker, over-segmented).
        CROSS_SIM_THRESHOLD = 0.75
        unmatched_labels = [l for l in label_embeddings if l not in identifications]
        matched_labels   = [l for l in label_embeddings if l in identifications]

        if unmatched_labels and matched_labels:
            logger.info(f"    [cross-dedup] Checking {len(unmatched_labels)} unmatched labels "
                        f"against {len(matched_labels)} matched labels...")
            for unmatched in unmatched_labels:
                emb_u = label_embeddings[unmatched].flatten()
                best_cross_sim  = -1.0
                best_cross_label = None
                for matched in matched_labels:
                    emb_m = label_embeddings[matched].flatten()
                    norm  = np.linalg.norm(emb_u) * np.linalg.norm(emb_m)
                    if norm == 0:
                        continue
                    sim = float(np.dot(emb_u, emb_m) / norm)
                    logger.info(f"      [cross-dedup] {unmatched} vs {matched}: sim={sim:.3f}")
                    if sim > best_cross_sim:
                        best_cross_sim   = sim
                        best_cross_label = matched

                if best_cross_label and best_cross_sim >= CROSS_SIM_THRESHOLD:
                    # Copy the matched label's identification
                    identifications[unmatched] = dict(identifications[best_cross_label])
                    logger.info(f"    [cross-dedup] ✅ {unmatched} → same person as {best_cross_label} "
                                f"({identifications[unmatched]['name']}, cross-sim={best_cross_sim:.3f})")
                else:
                    logger.info(f"    [cross-dedup] ❌ {unmatched} → genuinely different speaker "
                                f"(best cross-sim={best_cross_sim:.3f} < {CROSS_SIM_THRESHOLD})")

        return identifications

    def _embed_label(self, audio: np.ndarray, sr: int, segs: list) -> np.ndarray | None:
        """
        Extract a representative embedding for a speaker label.
        Averages multiple 3–12s windows for robustness.
        """
        WINDOW_MIN = int(3.0 * sr)
        WINDOW_MAX = int(12.0 * sr)

        chunks = []
        for seg in segs:
            start = int(seg['start'] * sr)
            end   = int(seg['end']   * sr)
            chunk = audio[start:end]
            dur   = len(chunk) / sr

            if dur < 3.0:
                continue

            if len(chunk) <= WINDOW_MAX:
                chunks.append(chunk)
            else:
                step = (len(chunk) - WINDOW_MAX) // 2
                for offset in [0, step, step * 2]:
                    window = chunk[offset:offset + WINDOW_MAX]
                    if len(window) >= WINDOW_MIN:
                        chunks.append(window)

        if not chunks:
            best = max(segs, key=lambda s: s['end'] - s['start'])
            start = int(best['start'] * sr)
            end   = min(int(best['end'] * sr), start + int(15.0 * sr))
            chunk = audio[start:end]
            if len(chunk) < WINDOW_MIN:
                return None
            chunks = [chunk]

        logger.info(f"      [embed] Averaging {len(chunks)} window(s) for robust embedding")

        embeddings = []
        for chunk in chunks[:5]:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    temp_path = tmp.name
                sf.write(temp_path, chunk, sr)
                url = self.embedding_service_url.rstrip("/") + "/embedding"

                with open(temp_path, "rb") as f:
                    files = {
                        "file": (
                            Path(temp_path).name,
                            f,
                            "audio/wav"
                        )
                    }
                    response = requests.post(url, files=files, timeout=(30, 300))

                response.raise_for_status()
                result = response.json()
                emb = np.asarray(result["embedding"], dtype=np.float32)
                embeddings.append(emb.flatten())
            except Exception as e:
                logger.warning(f"      [embed] ⚠️  window failed: {e}")
            finally:
                if temp_path:
                    try:
                        Path(temp_path).unlink()
                    except Exception:
                        pass

        if not embeddings:
            return None

        avg = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        return avg

    # ─────────────────────────────────────────────────────────────────────

    def _build_display_name_map(self, speaker_segments, raw_ids):
        """
        Build mapping from diarization label → display name.
        Unmatched labels get generic "Speaker N" names (numbered in order of appearance).
        """
        seen_labels = []
        seen_set    = set()
        for seg in sorted(speaker_segments, key=lambda s: float(s['start'])):
            label = seg['speaker']
            if label not in seen_set:
                seen_labels.append(label)
                seen_set.add(label)

        logger.info(f"    [names] Unique labels in time order: {seen_labels}")
        logger.info(f"    [names] raw_ids keys: {list(raw_ids.keys())}")

        display_map = {}
        next_number = 1

        for label in seen_labels:
            if label in raw_ids:
                enr_name = raw_ids[label]['name']
                enr_conf = raw_ids[label]['confidence']
                display_map[label] = enr_name
                logger.info(f"    [names] {label} → \"{enr_name}\" (enrolled, conf={enr_conf:.3f})")
            else:
                display_map[label] = f"Speaker {next_number}"
                next_number       += 1
                logger.info(f"    [names] {label} → \"{display_map[label]}\" (no enrolled match)")

        return display_map

    # ─────────────────────────────────────────────────────────────────────

    def _align_transcription(self, speaker_segments, transcription, display_map):
        all_words = []
        for seg in transcription:
            words = seg.get('words', [])
            if words:
                for w in words:
                    txt = w.get('word', '').strip()
                    if txt:
                        all_words.append({
                            'word':  txt,
                            'start': float(w.get('start', seg['start'])),
                            'end':   float(w.get('end',   seg['end'])),
                        })
            else:
                txt = seg.get('text', '').strip()
                if txt:
                    all_words.append({
                        'word':  txt,
                        'start': float(seg['start']),
                        'end':   float(seg['end']),
                    })

        logger.info(f"    [align] Total words from Whisper : {len(all_words)}")
        if not all_words:
            return []

        sorted_segs   = sorted(speaker_segments, key=lambda s: float(s['start']))
        word_assigned = [-1] * len(all_words)

        for seg_idx, seg in enumerate(sorted_segs):
            s, e = float(seg['start']), float(seg['end'])
            for w_idx, word in enumerate(all_words):
                mid = (word['start'] + word['end']) / 2.0
                if (s - 0.5) <= mid <= (e + 0.5):
                    word_assigned[w_idx] = seg_idx

        unassigned = [i for i, a in enumerate(word_assigned) if a == -1]
        logger.info(f"    [align] Unassigned after pass 1   : {len(unassigned)} words")
        for w_idx in unassigned:
            word = all_words[w_idx]
            for seg_idx, seg in enumerate(sorted_segs):
                s, e = float(seg['start']), float(seg['end'])
                if not (word['end'] <= s or word['start'] >= e):
                    word_assigned[w_idx] = seg_idx
                    break

        still_unassigned = [i for i, a in enumerate(word_assigned) if a == -1]
        if still_unassigned:
            logger.info(f"    [align] Fallback nearest-segment   : {len(still_unassigned)} words")
            for w_idx in still_unassigned:
                mid      = (all_words[w_idx]['start'] + all_words[w_idx]['end']) / 2.0
                best_idx = min(
                    range(len(sorted_segs)),
                    key=lambda i: abs(mid - (sorted_segs[i]['start'] + sorted_segs[i]['end']) / 2.0)
                )
                word_assigned[w_idx] = best_idx

        lost = sum(1 for a in word_assigned if a == -1)
        if lost:
            logger.error(f"    [align] ❌ CRITICAL: {lost} words could not be assigned!")
        else:
            logger.info(f"    [align] ✅ All {len(all_words)} words assigned to speakers")

        from collections import defaultdict
        seg_words = defaultdict(list)
        for w_idx, seg_idx in enumerate(word_assigned):
            if seg_idx >= 0:
                seg_words[seg_idx].append(all_words[w_idx])

        result = []
        for seg_idx, seg in enumerate(sorted_segs):
            words = seg_words.get(seg_idx, [])
            if not words:
                continue

            # Step 1: remove consecutive duplicate words
            clean_words = []
            prev_word   = None
            for w in words:
                word = w['word'].strip()
                if word and word.lower() != prev_word:
                    clean_words.append(word)
                prev_word = word.lower()

            text = " ".join(clean_words).strip()
            if not text:
                continue

            # Step 2: remove repeating phrase loops (Whisper hallucination)
            text = _remove_repeated_phrases_simple(text)
            text = _truncate_at_word_boundary(text, max_chars=2000)

            label        = seg['speaker']
            speaker_name = display_map.get(label, f"Speaker {label}")
            result.append({
                'start':              float(seg['start']),
                'end':                float(seg['end']),
                'speaker_label':      label,
                'speaker_name':       speaker_name,
                'speaker_confidence': 0.0,
                'text':               text,
            })
        return result

    def _extract_embedding(self, audio_path):
        emb = self.embedding_model(audio_path)
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        return emb.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Merge function — same-speaker only, with improved gap tolerance
# ─────────────────────────────────────────────────────────────────────────────
def merge_consecutive_speaker_segments(segments, max_gap=5.0):
    """
    FIX Issue 4: Increased default max_gap from 3.0 → 5.0s so that
    same-speaker segments with a short pause between them get merged.

    Also improved carry-over logic: a "Speaker N" segment that immediately
    follows a named speaker with a short gap is reassigned to the named speaker
    (handles diarization bleed-over at sentence boundaries).
    """
    if not segments:
        return segments

    # Step 1: Remove exact duplicates
    deduped = []
    for seg in segments:
        text_lower = seg['text'].strip().lower()
        recent = [s['text'].strip().lower() for s in deduped[-3:]]
        if text_lower not in recent:
            deduped.append(seg)
        else:
            logger.info(f"    [merge] 🗑️  Removed duplicate: {text_lower[:50]}")
    segments = deduped
    if not segments:
        return segments

    # ── Carry-over fix ───────────────────────────────────────────────────
    GENERIC_PREFIXES = ('Speaker ',)
    fixed = []
    for i, seg in enumerate(segments):
        spk = seg['speaker_name']
        is_generic = any(spk.startswith(p) for p in GENERIC_PREFIXES)
        if is_generic and fixed:
            prev = fixed[-1]
            gap = seg['start'] - prev['end']
            text = seg['text'].strip()
            connectors = ('and ', 'or ', 'but ', 'the ', 'to ', 'in ', 'with ', 'for ')
            mid_sentence = (
                text and text[0].islower()
            ) or any(text.lower().startswith(c) for c in connectors)
            prev_is_named = not any(prev['speaker_name'].startswith(p) for p in GENERIC_PREFIXES)
            if gap <= 2.0 and mid_sentence and prev_is_named:  # FIX: 1.5 → 2.0s
                logger.info(f"    [merge] 🔗 Carry-over: reassigning '{text[:40]}' "
                      f"from {spk} → {prev['speaker_name']}  (gap={gap:.2f}s, mid-sentence)")
                seg = seg.copy()
                seg['speaker_name'] = prev['speaker_name']
                seg['speaker_label'] = prev['speaker_label']
        fixed.append(seg)
    segments = fixed

    # Step 2: Merge same-speaker consecutive segments
    merged = []
    current = segments[0].copy()
    merges = 0

    for next_seg in segments[1:]:
        same_speaker = current['speaker_name'] == next_seg['speaker_name']
        time_gap = next_seg['start'] - current['end']

        if same_speaker and time_gap <= max_gap:
            current['text'] = current['text'].rstrip() + ' ' + next_seg['text'].lstrip()
            current['end'] = next_seg['end']
            merges += 1
        else:
            merged.append(current)
            current = next_seg.copy()

    merged.append(current)

    # Step 3: Drop very short segments
    final = [seg for seg in merged if len(seg['text'].split()) >= 1]
    dropped = len(merged) - len(final)
    logger.info(f"    [merge] {len(segments)} → {len(final)} segments ({merges} same-speaker merges, {dropped} short dropped)")

    from collections import Counter
    counts = Counter(s['speaker_name'] for s in final)
    logger.info(f"    [merge] Speaker breakdown after merge: {dict(counts)}")

    return final