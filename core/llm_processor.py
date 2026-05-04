#core/llm_processor.py
import re
import requests
import json
import time

from utils.logger import get_logger, log_exception

logger = get_logger("llm_processor")


_WORK_KEYWORDS = {
    'model', 'module', 'pipeline', 'api', 'mode', 'offline', 'online',
    'transcri', 'diariz', 'speaker', 'noise', 'filter', 'whisper', 'llm',
    'embedding', 'vector', 'threshold', 'segment', 'audio', 'video', 'wav',
    'mp3', 'mp4', 'file', 'upload', 'download', 'database', 'server',
    'deploy', 'integrat', 'implement', 'develop', 'build', 'code', 'script',
    'python', 'torch', 'cuda', 'gpu', 'cpu', 'rnn', 'deep', 'neural',
    'conformer', 'pyannote', 'streamlit', 'interface', 'output', 'input',
    'summar', 'agenda', 'task', 'assign', 'deadline', 'priorit', 'feature',
    'function', 'class', 'method', 'test', 'debug', 'error', 'fix', 'bug',
    'version', 'update', 'review', 'merge', 'branch', 'commit', 'push',
    'data', 'dataset', 'train', 'inference', 'accuracy', 'result', 'report',
    'meeting', 'project', 'team', 'work', 'plan', 'discuss', 'point',
    'step', 'phase', 'stage', 'process', 'system', 'framework', 'tool',
    'karan', 'manan', 'bhautik', 'priyanshi', 'pautik', 'poutik',
}

_SHORT_SEGMENT_WORD_LIMIT = 8


def _is_filler_sentence(text: str) -> bool:
    words = text.lower().split()
    if len(words) > _SHORT_SEGMENT_WORD_LIMIT:
        return False
    text_lower = text.lower()
    for kw in _WORK_KEYWORDS:
        if kw in text_lower:
            return False
    return True


class QwenProcessor:
    def __init__(self, config):
        self.config      = config
        self.base_url    = config['llm']['base_url']
        self.model       = config['llm']['model']
        self.temperature = config['llm']['temperature']
        self.max_tokens  = config['llm']['max_tokens']
        self.MAX_TRANSCRIPT_CHARS = 20_000

    # ─────────────────────────────────────────────────────────────────────

    def clean_transcript(self, segments):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  [STEP 3.5] TRANSCRIPT CLEANING (LLM)")
        logger.info(f"  Segments in : {len(segments)}")
        logger.info(f"  LLM model   : {self.model}")
        logger.info(f"{'─'*60}")
        t_total = time.time()

        if not segments:
            logger.info(f"  [STEP 3.5] No segments to clean, skipping")
            return segments

        pre_filtered = []
        for seg in segments:
            text = seg.get('text', '').strip()
            if _is_filler_sentence(text):
                logger.info(f"    [pre-filter] 🗑️  Filler removed: '{text}'")
            else:
                pre_filtered.append(seg)
        if len(pre_filtered) < len(segments):
            logger.info(f"    [pre-filter] Removed {len(segments) - len(pre_filtered)} filler segment(s)")
        segments = pre_filtered

        if not segments:
            logger.info(f"  [STEP 3.5] No segments remain after pre-filter")
            return segments

        lines = []
        for i, seg in enumerate(segments):
            speaker = seg.get('speaker_name', 'Unknown')
            text    = seg.get('text', '').strip()
            lines.append(f"[{i}] {speaker}: {text}")

        numbered_transcript = "\n".join(lines)
        total_chars = len(numbered_transcript)
        logger.info(f"  [STEP 3.5] Transcript characters sent to LLM: {total_chars}")

        prompt = f"""You are a meeting transcript cleaner. Your task is to remove ONLY clearly off-topic, non-work content.

CRITICAL RULE: This is a technical/business meeting. ALL work-related content MUST be kept, including:
- Technical discussions, implementation details, progress updates
- Project architecture, tools, models being discussed
- Status reports from any team member
- Questions, answers, or clarifications about the project
- Any mention of tasks, assignments, deadlines, or responsibilities
- Meeting agenda and agenda items
- Decisions, action items, or plans

ONLY REMOVE a segment if it is CLEARLY and ENTIRELY one of these:
1. Pure greetings with no work content ("hello everyone", "good morning", "bye bye")
2. Pure personal small talk completely unrelated to work ("how was your weekend?", "are you feeling better?")
3. Break/logistics announcements with no work content ("let's take a break", "bathroom break")
4. Mentions of tea, coffee, food, or refreshments ("tea hasn't come yet", "chai lao", "bring tea", "get coffee")
5. Audio-only technical issues with no content ("can you hear me?", "you're muted", "bad network")
6. Single filler sounds with no content ("hmm", "ok ok ok", "haan haan haan")

WHEN IN DOUBT → KEEP THE SEGMENT.

CRITICAL TEXT RULE: You have exactly TWO choices for each segment:
  a) KEEP it — copy the text EXACTLY as given, character for character, no changes at all
  b) REMOVE it completely — put its index in "removed"
You are STRICTLY FORBIDDEN from partially editing, rewording, shortening, or modifying any segment text.
The ONLY exception is replacing a confidential value (password, bank number, bank account, IFSC etc.) with [REDACTED].

TRANSCRIPT:
{numbered_transcript}

Return ONLY ONE valid JSON object. No explanation, no markdown, no code fences.

VALID EXAMPLE FORMAT:
{{"keep":[{{"index":0,"text":"So, hello, good afternoon everyone."}},{{"index":2,"text":"usually, directly we can assign it"}}],"removed":[1]}}

RULES:
- Default is to KEEP everything. Only put an index in "removed" if 100% off-topic.
- In "keep", the "text" value MUST be an exact copy of the original.
- Indices in "removed": only greetings, tea/food/drink mentions, personal chat, breaks, filler sounds.

JSON:"""

        logger.info(f"  [STEP 3.5] Sending to LLM...")
        t_llm = time.time()
        response = self._call_llm(prompt, max_tokens_override=2500)
        llm_elapsed = time.time() - t_llm
        logger.info(f"  [STEP 3.5] LLM response received in {llm_elapsed:.1f}s")

        try:
            result = self._extract_json(response)

            keep_map = {item['index']: item['text'] for item in result.get('keep', [])}
            removed  = result.get('removed', [])

            speaker_segment_map = {}
            for i, seg in enumerate(segments):
                speaker = seg.get('speaker_name', 'Unknown')
                if speaker not in speaker_segment_map:
                    speaker_segment_map[speaker] = []
                speaker_segment_map[speaker].append(i)

            removed_set = set(removed)
            for speaker, indices in speaker_segment_map.items():
                if all(idx in removed_set for idx in indices):
                    longest_idx = max(
                        indices,
                        key=lambda idx: len(segments[idx].get('text', ''))
                    )
                    removed_set.discard(longest_idx)
                    logger.warning(f"    [clean] ⚠️  SAFEGUARD: Restored [{longest_idx}] for '{speaker}' "
                          f"— cannot remove all segments for a speaker")
            removed = list(removed_set)

            cleaned_segments = []
            redacted_count   = 0
            for i, seg in enumerate(segments):
                if i in removed:
                    logger.info(f"    [clean] 🗑️  Removed [{i}] {seg.get('speaker_name','?')}: "
                          f"{seg.get('text','')[:60]}...")
                    continue
                new_seg = seg.copy()
                if i in keep_map:
                    llm_text = keep_map[i]
                    if '[REDACTED]' in llm_text:
                        new_seg['text'] = llm_text
                        logger.info(f"    [clean] 🔒 Redacted [{i}]")
                        redacted_count += 1
                    else:
                        new_seg['text'] = seg.get('text', '').strip()
                else:
                    new_seg['text'] = seg.get('text', '').strip()
                cleaned_segments.append(new_seg)

            removal_rate = len(removed) / len(segments) if segments else 0
            if removal_rate > 0.60:
                logger.warning(f"    [clean] ⚠️  SAFEGUARD: LLM removed {removal_rate*100:.0f}% of segments "
                      f"— too aggressive. Reverting to original transcript.")
                cleaned_segments = segments
                removed          = []

            total_elapsed = time.time() - t_total
            logger.info(f"  [STEP 3.5] ✓ CLEANING COMPLETE")
            logger.info(f"             Input    : {len(segments)} segments")
            logger.info(f"             Output   : {len(cleaned_segments)} segments")
            logger.info(f"             Removed  : {len(removed)}")
            logger.info(f"             Redacted : {redacted_count}")
            logger.info(f"             Time     : {total_elapsed:.1f}s  (LLM: {llm_elapsed:.1f}s)")
            logger.info(f"{'─'*60}\n")
            return cleaned_segments

        except Exception as e:
            logger.warning(f"  [STEP 3.5] ⚠️  Parsing failed ({e}), using original segments")
            logger.info(f"{'─'*60}\n")
            return segments

    # ─────────────────────────────────────────────────────────────────────

    def generate_summary(self, identified_segments, enrolled_speakers=None):
        logger.info(f"\n{'─'*60}")
        logger.info(f"  [STEP 4] MEETING SUMMARY (LLM)")
        logger.info(f"  Segments in  : {len(identified_segments)}")
        logger.info(f"  LLM model    : {self.model}")
        logger.info(f"{'─'*60}")
        t_total = time.time()

        enrolled_names = []
        if enrolled_speakers:
            enrolled_names = [
                {"name": s["name"], "role": s.get("role", "")}
                for s in enrolled_speakers
            ]
            logger.info(f"  [STEP 4] Enrolled speakers injected: "
                        f"{[n['name'] for n in enrolled_names]}")

        # Collect all speaker label names from transcript (Speaker 1, Speaker 2, etc.)
        transcript_speaker_names = sorted(set(
            seg.get('speaker_name', '') for seg in identified_segments
            if seg.get('speaker_name', '')
        ))
        logger.info(f"  [STEP 4] Transcript speaker names: {transcript_speaker_names}")

        transcript = self._build_transcript(identified_segments)
        char_count = len(transcript)
        logger.info(f"  [STEP 4] Transcript characters: {char_count}  (limit: {self.MAX_TRANSCRIPT_CHARS})")

        if char_count <= self.MAX_TRANSCRIPT_CHARS:
            logger.info(f"  [STEP 4] Short meeting — single-shot LLM call")
            prompt    = self._create_summary_prompt(transcript, enrolled_names, transcript_speaker_names)
            t_llm     = time.time()
            response  = self._call_llm(prompt, max_tokens_override=1500)
            llm_time  = time.time() - t_llm
            logger.info(f"  [STEP 4] LLM responded in {llm_time:.1f}s")
            result = response if isinstance(response, dict) else self._parse_summary(response)
        else:
            logger.info(f"  [STEP 4] Long meeting — chunked processing")
            result = self._process_long_transcript(transcript, identified_segments, enrolled_names, transcript_speaker_names)

        # Post-process: fix names in tasks using description text
        result = self._fix_task_names(result, enrolled_names, transcript_speaker_names)

        total_elapsed = time.time() - t_total
        logger.info(f"  [STEP 4] ✓ SUMMARY COMPLETE — total: {total_elapsed:.1f}s")
        logger.info(f"           Key points : {len(result.get('key_points', []))}")
        logger.info(f"           Decisions  : {len(result.get('decisions', []))}")
        logger.info(f"           Tasks      : {len(result.get('tasks', []))}")
        logger.info(f"{'─'*60}\n")
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Core fix: extract real names from description when LLM writes "Speaker N"
    # ─────────────────────────────────────────────────────────────────────

    def _fix_task_names(self, result, enrolled_names, transcript_speaker_names):
        """
        Two-stage fix for task assignee/assigner fields:

        STAGE 1 — Name extraction from description:
          When the LLM puts "Speaker 1" or "Speaker N" as assignee/assigner,
          it means it correctly identified the task but failed to resolve the
          speaker label to a real name. We scan the task's description text
          for any enrolled names (or known name variants) and use those instead.

          Example: description = "Karan and Manan both are working on that offline mode."
                   assignee = "Speaker 1"  →  we find "Karan" in description → set assignee = "Karan"
                   For multi-person tasks we keep the first name found as primary assignee.

        STAGE 2 — Fuzzy name correction:
          For any name that IS present but looks garbled (e.g. "Pavan Bhut" vs "Bhautik"),
          we snap it to the closest canonical name using token overlap.
        """
        # Build the full set of valid canonical names (enrolled + generic speaker labels)
        enrolled_name_list = [e['name'].strip() for e in enrolled_names]
        valid_names = set(enrolled_name_list)
        for name in transcript_speaker_names:
            valid_names.add(name.strip())

        # Build a lookup: lowercased first token of each enrolled name → full name
        # e.g. "karan soni" → first token "karan" maps to "Karan Soni"
        # This lets us find names mentioned by first name only in descriptions
        first_name_map = {}
        for name in enrolled_name_list:
            tokens = name.lower().split()
            for token in tokens:
                if len(token) >= 3:  # skip very short tokens
                    if token not in first_name_map:
                        first_name_map[token] = name
                    # If multiple enrolled names share a first-name token, keep both
                    # by storing a list
            # Also map full lowercase name
            first_name_map[name.lower()] = name

        logger.info(f"    [fix-names] first_name_map: {first_name_map}")

        def _is_generic_speaker(name: str) -> bool:
            """Returns True if name is a generic label like 'Speaker 1', 'Speaker 2'"""
            if not name:
                return True
            stripped = name.strip().lower()
            return re.match(r'^speaker\s*\d+$', stripped) is not None

        def _extract_names_from_description(desc: str) -> list[str]:
            """
            Scan description text for enrolled names (by first name or full name).
            Returns list of matched canonical names in order of appearance.
            """
            if not desc:
                return []
            desc_lower = desc.lower()
            found = []
            seen  = set()
            # Split description into tokens (words)
            tokens = re.findall(r"[a-zA-Z']+", desc_lower)
            for token in tokens:
                if token in first_name_map:
                    canonical = first_name_map[token]
                    if canonical not in seen:
                        found.append(canonical)
                        seen.add(canonical)
            return found

        def _snap_to_canonical(raw_name: str) -> str:
            """Fuzzy-snap a name to the closest canonical name."""
            if not raw_name or raw_name.strip() in ('Unknown', 'Not specified', ''):
                return raw_name
            raw_stripped = raw_name.strip()
            if raw_stripped in valid_names:
                return raw_stripped
            raw_lower = raw_stripped.lower()
            # Exact case-insensitive
            for v in valid_names:
                if v.lower() == raw_lower:
                    return v
            # Token overlap
            raw_tokens = set(raw_lower.split())
            best_name  = None
            best_score = 0
            for v in valid_names:
                v_tokens = set(v.lower().split())
                overlap  = len(raw_tokens & v_tokens)
                substr   = 1 if (raw_lower in v.lower() or v.lower() in raw_lower) else 0
                score    = overlap + substr
                if score > best_score:
                    best_score = score
                    best_name  = v
            if best_score >= 1 and best_name:
                logger.info(f"    [fix-names] fuzzy: '{raw_stripped}' → '{best_name}' (score={best_score})")
                return best_name
            logger.warning(f"    [fix-names] ⚠️  '{raw_stripped}' did not match any known name — keeping")
            return raw_stripped

        tasks = result.get('tasks', [])
        for task in tasks:
            description  = task.get('description', '')
            orig_assignee = task.get('assignee', '')
            orig_assigner = task.get('assigner', '')

            # ── Stage 1: Replace generic "Speaker N" from description ──
            if _is_generic_speaker(orig_assignee):
                names_in_desc = _extract_names_from_description(description)
                if names_in_desc:
                    # For assignee: use first name found in description
                    # (the person being assigned the work)
                    task['assignee'] = names_in_desc[0]
                    logger.info(f"    [fix-names] assignee: '{orig_assignee}' → '{task['assignee']}' "
                                f"(extracted from description)")
                else:
                    # No name found in description — leave as generic
                    logger.info(f"    [fix-names] assignee: '{orig_assignee}' → no name found in description, keeping")

            if _is_generic_speaker(orig_assigner):
                names_in_desc = _extract_names_from_description(description)
                if names_in_desc:
                    # For assigner: if multiple names in desc, the assigner is typically
                    # the person giving the task, not doing it. But when the description
                    # is a summary like "Karan and Manan both are working on offline mode",
                    # there's no explicit assigner — keep the speaker label or use last name
                    # as a heuristic if only one name exists.
                    if len(names_in_desc) == 1:
                        task['assigner'] = names_in_desc[0]
                        logger.info(f"    [fix-names] assigner: '{orig_assigner}' → '{task['assigner']}' "
                                    f"(only one name in description)")
                    else:
                        # Multiple names — assigner is likely whoever said the summary sentence.
                        # We can't know without segment context, so keep generic.
                        logger.info(f"    [fix-names] assigner: '{orig_assigner}' → keeping (multiple names in desc, ambiguous)")
                else:
                    logger.info(f"    [fix-names] assigner: '{orig_assigner}' → no name found in description, keeping")

            # ── Stage 2: Fuzzy-fix any non-generic garbled names ──
            if not _is_generic_speaker(task['assignee']):
                fixed = _snap_to_canonical(task['assignee'])
                if fixed != task['assignee']:
                    logger.info(f"    [fix-names] assignee fuzzy-fixed: '{task['assignee']}' → '{fixed}'")
                    task['assignee'] = fixed

            if not _is_generic_speaker(task['assigner']):
                fixed = _snap_to_canonical(task['assigner'])
                if fixed != task['assigner']:
                    logger.info(f"    [fix-names] assigner fuzzy-fixed: '{task['assigner']}' → '{fixed}'")
                    task['assigner'] = fixed

        result['tasks'] = tasks
        return result

    # ─────────────────────────────────────────────────────────────────────

    def _build_transcript(self, segments):
        lines = []
        for seg in segments:
            speaker   = seg.get('speaker_name', 'Unknown')
            text      = seg.get('text', '').strip()
            start     = seg.get('start', 0)
            timestamp = f"[{int(start // 60):02d}:{int(start % 60):02d}]"
            if text:
                lines.append(f"{timestamp} {speaker}: {text}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────

    def _process_long_transcript(self, transcript, segments, enrolled_names=None, transcript_speaker_names=None):
        lines   = transcript.split('\n')
        chunks  = []
        current_chunk = []
        current_len   = 0
        for line in lines:
            if current_len + len(line) + 1 > self.MAX_TRANSCRIPT_CHARS and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_len   = len(line)
            else:
                current_chunk.append(line)
                current_len += len(line) + 1
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        logger.info(f"  [STEP 4] Split into {len(chunks)} chunks")

        partial_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  [STEP 4] Processing chunk {i+1}/{len(chunks)}...")
            t_chunk  = time.time()
            prompt   = self._create_partial_prompt(chunk, i + 1, len(chunks), enrolled_names, transcript_speaker_names)
            response = self._call_llm(prompt, max_tokens_override=1200)
            parsed   = response if (isinstance(response, dict) and 'overview' in response) else self._parse_summary(response)
            logger.info(f"  [STEP 4] Chunk {i+1} done in {time.time()-t_chunk:.1f}s")
            partial_summaries.append(parsed)

        return self._merge_summaries(partial_summaries)

    def _merge_summaries(self, summaries):
        all_key_points = []
        all_decisions  = []
        all_tasks      = []
        overviews      = []
        for s in summaries:
            if s.get('overview') and s['overview'] not in ('', 'N/A'):
                overviews.append(s['overview'])
            all_key_points.extend(s.get('key_points', []))
            all_decisions.extend(s.get('decisions', []))
            all_tasks.extend(s.get('tasks', []))

        def dedupe(lst):
            seen, out = set(), []
            for item in lst:
                key = item if isinstance(item, str) else str(item)
                if key not in seen:
                    seen.add(key)
                    out.append(item)
            return out

        merged_overview = " ".join(overviews) if overviews else "Multi-part meeting summary."
        if len(merged_overview) > 600:
            t_compress = time.time()
            compress_prompt = (
                f"Summarize the following into 2-3 concise sentences:\n\n{merged_overview}\n\n"
                "Return only the summary text, nothing else."
            )
            compressed = self._call_llm(compress_prompt, max_tokens_override=300)
            if isinstance(compressed, str) and len(compressed) > 10:
                merged_overview = compressed.strip()
            logger.info(f"  [STEP 4] Overview compression done in {time.time()-t_compress:.1f}s")

        return {
            'overview':   merged_overview,
            'key_points': dedupe(all_key_points),
            'decisions':  dedupe(all_decisions),
            'tasks':      all_tasks,
        }

    # ─────────────────────────────────────────────────────────────────────

    def _create_summary_prompt(self, transcript, enrolled_names=None, transcript_speaker_names=None):
        # Build canonical names block combining enrolled + generic labels
        all_known_names = []
        enrolled_name_set = set()
        if enrolled_names:
            for e in enrolled_names:
                all_known_names.append(
                    f"  - {e['name']}" + (f" ({e['role']})" if e['role'] else "")
                )
                enrolled_name_set.add(e['name'])
        if transcript_speaker_names:
            for name in transcript_speaker_names:
                if name not in enrolled_name_set:
                    all_known_names.append(f"  - {name} (unidentified speaker)")

        if all_known_names:
            name_lines = "\n".join(all_known_names)
            speaker_block = (
                f"\nCANONICAL SPEAKER NAMES (the only valid names in this meeting):\n"
                f"{name_lines}\n\n"
                f"IMPORTANT NAME RULES:\n"
                f"1. For assignee/assigner fields: use the ACTUAL PERSON'S NAME mentioned in the task description, not the speaker label.\n"
                f"   Example: if description says 'Karan and Manan are working on offline mode', assignee = 'Karan' (or 'Manan'), NOT 'Speaker 1'.\n"
                f"2. Never invent names. Only use names explicitly mentioned in the transcript text.\n"
                f"3. If a name in the transcript sounds like a garbled version of a canonical name, use the canonical name.\n"
                f"4. Speaker labels like 'Speaker 1' should only appear in assignee/assigner if no real name is mentioned.\n"
            )
        else:
            speaker_block = ""

        return f"""You are an AI assistant analyzing a meeting transcript. Provide a factual summary using ONLY what is explicitly said.
{speaker_block}
MEETING TRANSCRIPT:
{transcript}

Return ONLY valid JSON with no explanation, no markdown, no code fences. Start your response with {{ and end with }}.

FORMAT:
{{"overview":"2-3 sentence overview","key_points":["point 1","point 2"],"decisions":["decision 1"],"tasks":[{{"title":"Short title","description":"Exact speaker words showing assignment","assignee":"The person doing the work (use their actual name from description, not speaker label)","assigner":"The person giving the task (use their actual name, not speaker label)","deadline":"Deadline or Not specified","priority":"High/Medium/Low"}}]}}

═══ TASK EXTRACTION RULES ═══

Include a task ONLY if one of these 3 patterns exists:

PATTERN A — Direct assignment:
  "can you X", "you need to X", "please do X", "your task is X", "you handle X"
  → assignee = the person being asked, assigner = the speaker

PATTERN B — Named person stated as working on area:
  "Karan is working on Y", "Manan will handle Y"
  → assignee = Karan (or Manan), assigner = whoever stated this

PATTERN C — End-of-meeting agenda summary naming who does what:
  "Karan and Manan both are working on that offline mode. And that Pautik and Priyansi which are working on that online mode."
  → Task 1: title="Work on offline mode", assignee="Karan", description=exact sentence
  → Task 2: title="Work on online mode", assignee="Pautik", description=exact sentence

KEY RULE FOR ASSIGNEE: Look at the task description text. Extract the PERSON'S NAME mentioned there.
Do NOT put the speaker label (Speaker 1, Speaker 2) as assignee if a real name appears in the description.

DESCRIPTION FIELD: Copy ONLY the exact words from the transcript. Do NOT add, expand, or infer.

If no tasks found: "tasks":[]

JSON:"""

    def _create_partial_prompt(self, chunk, part_num, total_parts, enrolled_names=None, transcript_speaker_names=None):
        all_known_names = []
        enrolled_name_set = set()
        if enrolled_names:
            for e in enrolled_names:
                all_known_names.append(
                    f"  - {e['name']}" + (f" ({e['role']})" if e['role'] else "")
                )
                enrolled_name_set.add(e['name'])
        if transcript_speaker_names:
            for name in transcript_speaker_names:
                if name not in enrolled_name_set:
                    all_known_names.append(f"  - {name} (unidentified speaker)")

        if all_known_names:
            name_lines = "\n".join(all_known_names)
            speaker_block = (
                f"\nCANONICAL NAMES:\n{name_lines}\n"
                f"RULE: assignee/assigner must be the PERSON NAMED IN THE DESCRIPTION, not the speaker label.\n"
            )
        else:
            speaker_block = ""

        return f"""You are analyzing part {part_num} of {total_parts} of a meeting transcript.
{speaker_block}

TRANSCRIPT PART {part_num}/{total_parts}:
{chunk}

Return ONLY valid JSON. Start with {{ end with }}.

{{"overview":"Brief summary","key_points":["point 1"],"decisions":["decision 1"],"tasks":[{{"title":"Short title","description":"Exact speaker words","assignee":"Person named in description (not speaker label)","assigner":"Person giving task (not speaker label)","deadline":"Not specified","priority":"Medium"}}]}}

TASK RULE: include only if direct assignment or agenda summary names who does what.
ASSIGNEE: extract from description text — the person DOING the work.
DESCRIPTION: exact sentence(s) from transcript only.
If no tasks: "tasks":[]

JSON:"""

    # ─────────────────────────────────────────────────────────────────────

    def _extract_json(self, response: str) -> dict:
        if isinstance(response, dict):
            return response

        text = response if isinstance(response, str) else str(response)

        if "</think>" in text:
            text = text.split("</think>", 1)[-1]
        if "<think>" in text:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = text.replace("<think>", "")

        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        text = text.strip()

        json_text = self._extract_balanced_json(text)
        if not json_text:
            json_text = text

        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)

        logger.debug("\n" + "=" * 80)
        logger.debug("EXTRACTED JSON (first 2000 chars)")
        logger.debug("=" * 80)
        logger.debug(json_text[:2000])
        logger.debug("=" * 80 + "\n")

        return json.loads(json_text)

    def _extract_balanced_json(self, text: str) -> str:
        start = text.find('{')
        if start == -1:
            return ''
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return ''

    # ─────────────────────────────────────────────────────────────────────

    def _call_llm(self, prompt, max_tokens_override=None):
        url = f"{self.base_url}/chat/completions"
        effective_tokens = max_tokens_override or self.max_tokens

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON API. "
                        "Output ONLY a single valid JSON object. "
                        "No thinking, no explanation, no markdown, no prose. "
                        "Start your response with { and end with }."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": effective_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        try:
            logger.info(
                f"    [llm] POST {url}  "
                f"(max_tokens={effective_tokens}, temp={self.temperature}, thinking=off)"
            )

            t = time.time()
            response = requests.post(url, json=payload, timeout=(30, 900))

            if not response.ok:
                logger.error(f"    [llm] ❌ Status Code: {response.status_code}")
                logger.error(f"    [llm] ❌ Response Body: {response.text[:500]}")

            response.raise_for_status()

            raw_content = response.json()['choices'][0]['message']['content']

            content = raw_content
            if "</think>" in content:
                content = content.split("</think>", 1)[-1].strip()
            if "<think>" in content:
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = content.replace("<think>", "").strip()

            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            content = content.strip()

            logger.info(f"    [llm] ✓ Response in {time.time()-t:.1f}s  ({len(content)} chars)")
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"    [llm] ❌ LLM request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"    [llm] ❌ Response: {e.response.text[:500]}")
            return self._get_fallback_summary()

        except Exception as e:
            logger.error(f"    [llm] ❌ Unexpected error: {e}")
            log_exception(logger, "Unexpected LLM error")
            return self._get_fallback_summary()

    # ─────────────────────────────────────────────────────────────────────

    def _parse_summary(self, response):
        if isinstance(response, dict):
            return response
        if isinstance(response, str):
            try:
                return self._extract_json(response)
            except Exception as e:
                logger.warning(f"    [llm] ⚠️ Could not parse response — {e}")
        return self._get_fallback_summary()

    def _get_fallback_summary(self):
        return {
            "overview": (
                f"Unable to generate summary. "
                f"LLM may be unavailable. Check: {self.base_url}"
            ),
            "key_points": [
                "LLM connection failed",
                f"Verify server is running at {self.base_url}"
            ],
            "decisions": [],
            "tasks": [],
        }