"""
Meeting Agent – FastAPI Gateway
================================
Replaces the Streamlit app.py with a fully async REST API.
Every endpoint emits structured JSON logs + SSE progress events.

IMPORTANT: speechbrain is stubbed at the top of this file so that
pyannote.audio can be imported without the k2 module being present.
"""

# ── Stub speechbrain.integrations.k2_fsa BEFORE any other import ──────────
import sys
import types

def _stub_k2():
    stub_names = [
        "speechbrain",
        "speechbrain.utils",
        "speechbrain.utils.importutils",
        "speechbrain.integrations",
        "speechbrain.integrations.k2_fsa",
    ]
    for name in stub_names:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    k2_mod = sys.modules["speechbrain.integrations.k2_fsa"]
    k2_mod.__all__ = []

_stub_k2()
# ──────────────────────────────────────────────────────────────────────────

import asyncio
import json
import os
import time
import tempfile
import traceback
import uuid
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import imageio_ffmpeg as ffmpeg_lib
import numpy as np
import soundfile as sf
import yaml
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

warnings.filterwarnings("ignore")

# ── FFmpeg bootstrap ───────────────────────────────────────────────────────
ffmpeg_path = ffmpeg_lib.get_ffmpeg_exe()
ffmpeg_dir  = os.path.dirname(ffmpeg_path)
os.environ["PATH"]               = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

from pydub import AudioSegment
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg    = ffmpeg_path
AudioSegment.ffprobe   = ffmpeg_path.replace("ffmpeg", "ffprobe")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"]         = "1"

# ── Config ─────────────────────────────────────────────────────────────────
# config.yaml sits in the same folder as api.py (project root)
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    CONFIG = yaml.safe_load(_f)

from utils.logger import get_logger, log_exception

logger = get_logger("api")

# ── In-memory job store ────────────────────────────────────────────────────
JOBS: dict[str, dict] = {}

# ── Numpy JSON encoder ─────────────────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def _json(obj) -> str:
    return json.dumps(obj, cls=NumpyEncoder)


# ── Component registry ────────────────────────────────────────────────────
class ComponentRegistry:
    enrollment    = None
    diarization   = None
    transcriber   = None
    llm           = None
    identifier    = None
    noise_reducer = None
    voice_db      = None

    @classmethod
    def load(cls):
        from core.diarization         import SpeakerDiarization
        from core.llm_processor       import QwenProcessor
        from core.noise_reduction     import NoiseReducer
        from core.speaker_identifier  import SpeakerIdentifier
        from core.transcription       import WhisperTranscriber
        from core.voice_enrollment    import VoiceEnrollment
        from database.voice_database  import VoiceDatabase

        t_all = time.time()
        log_banner("INITIALIZING ALL COMPONENTS")

        cls.voice_db      = VoiceDatabase(CONFIG["paths"]["speaker_embeddings"])
        cls.enrollment    = VoiceEnrollment(CONFIG)
        cls.diarization   = SpeakerDiarization(CONFIG)
        cls.transcriber   = WhisperTranscriber(CONFIG)
        cls.llm           = QwenProcessor(CONFIG)
        cls.identifier    = SpeakerIdentifier(CONFIG, cls.voice_db)
        cls.noise_reducer = NoiseReducer()

        log_banner(f"ALL COMPONENTS READY  ({time.time()-t_all:.1f}s)")


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    ComponentRegistry.load()
    yield
    log_section("SHUTDOWN — Cleaning up resources")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Meeting Agent API",
    version="2.0.0",
    description="Offline AI meeting summarizer — FastAPI backend for Tauri frontend",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def log_banner(msg: str):
    bar = "#" * 72
    logger.info(f"\n{bar}\n#  {msg}\n{bar}\n")

def log_section(msg: str):
    logger.info(f"\n{'─'*60}\n  {msg}\n{'─'*60}")

def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    logger.info(f"  [{ts}] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Job helpers
# ─────────────────────────────────────────────────────────────────────────────
def new_job(job_id: str, name: str) -> dict:
    job = {
        "id":           job_id,
        "name":         name,
        "status":       "pending",
        "progress":     0,
        "current_step": "",
        "steps":        [],
        "result":       None,
        "error":        None,
        "created_at":   time.time(),
    }
    JOBS[job_id] = job
    return job


def job_step(job: dict, step: str, message: str, elapsed: float, progress: int):
    entry = {
        "step":    step,
        "message": message,
        "elapsed": round(elapsed, 2),
        "ts":      time.strftime("%H:%M:%S"),
    }
    job["steps"].append(entry)
    job["progress"] = progress
    log(f"[{step}] {message}  ({elapsed:.1f}s)  progress={progress}%")


def get_fresh_db():
    from database.voice_database import VoiceDatabase
    return VoiceDatabase(CONFIG["paths"]["speaker_embeddings"])


# ─────────────────────────────────────────────────────────────────────────────
# MP4 → WAV
# ─────────────────────────────────────────────────────────────────────────────
def convert_mp4_to_wav(mp4_path: str) -> str:
    import subprocess
    log(f"[CONVERT] MP4 → WAV: {Path(mp4_path).name}")
    t        = time.time()
    wav_path = mp4_path.replace(".mp4", ".wav")
    cmd      = [
        ffmpeg_path, "-i", mp4_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1", "-y", wav_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed:\n{result.stderr}")
    log(f"[CONVERT] ✓ Done in {time.time()-t:.2f}s → {Path(wav_path).name}")
    return wav_path


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline timing summary
# ─────────────────────────────────────────────────────────────────────────────
def _print_timing_summary(steps: list, audio_duration: float, total_elapsed: float):
    bar = "═" * 72
    rtf = total_elapsed / audio_duration if audio_duration > 0 else 0
    logger.info(f"\n\n{bar}")
    logger.info(f"  📊  PIPELINE PERFORMANCE SUMMARY")
    logger.info(f"{bar}")
    logger.info(f"  Audio duration      : {audio_duration/60:.1f} min  ({audio_duration:.0f}s)")
    logger.info(f"  Total pipeline time : {total_elapsed/60:.1f} min  ({total_elapsed:.0f}s)")
    logger.info(f"  Real-time ratio     : {rtf:.2f}x")
    logger.info(f"{bar}")
    for s in steps:
        pct   = s["elapsed"] / total_elapsed * 100 if total_elapsed > 0 else 0
        t_str = f"{s['elapsed']/60:.1f}m" if s["elapsed"] >= 60 else f"{s['elapsed']:.1f}s"
        logger.info(f"  {s['step']:<38}  {t_str:>8}  {pct:>6.1f}%")
    logger.info(f"{bar}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Background pipeline
# ─────────────────────────────────────────────────────────────────────────────
async def run_pipeline(job_id: str, recording_path: Path, filename: str):
    job = JOBS[job_id]
    job["status"] = "running"

    try:
        audio_info     = sf.info(str(recording_path))
        audio_duration = audio_info.duration

        log_banner(f"PIPELINE START  |  job={job_id}  |  file={filename}")
        loop = asyncio.get_event_loop()

        # STEP 0 : Noise Reduction
        job["current_step"] = "noise_reduction"
        t0 = time.time()
        recording_path = Path(await loop.run_in_executor(
            None, ComponentRegistry.noise_reducer.enhance_audio, str(recording_path)
        ))
        job_step(job, "noise_reduction", f"DeepFilterNet complete → {recording_path.name}", time.time()-t0, 15)

        # STEP 1 : Diarization
        job["current_step"] = "diarization"
        t1 = time.time()
        speaker_segments = await loop.run_in_executor(
            None, ComponentRegistry.diarization.diarize, str(recording_path)
        )
        unique_speakers = len(set(s["speaker"] for s in speaker_segments))
        job_step(job, "diarization", f"{unique_speakers} speakers, {len(speaker_segments)} segments", time.time()-t1, 35)

        # STEP 2 : Transcription
        job["current_step"] = "transcription"
        t2 = time.time()
        transcription = await loop.run_in_executor(
            None, ComponentRegistry.transcriber.transcribe, str(recording_path)
        )
        total_words = sum(len(s.get("words", [])) for s in transcription)
        t2_elapsed  = time.time() - t2
        job_step(job, "transcription", f"{len(transcription)} segments, {total_words} words", t2_elapsed, 55)

        # STEP 3 : Speaker Identification
        job["current_step"] = "speaker_identification"
        ComponentRegistry.identifier.voice_db = get_fresh_db()
        t3 = time.time()
        identified_segments = await loop.run_in_executor(
            None, ComponentRegistry.identifier.identify_speakers,
            str(recording_path), speaker_segments, transcription,
        )
        job_step(job, "speaker_identification", f"{len(identified_segments)} aligned segments", time.time()-t3, 70)

        # STEP 3.5 : Transcript Cleaning
        job["current_step"] = "transcript_cleaning"
        t35 = time.time()
        identified_segments = await loop.run_in_executor(
            None, ComponentRegistry.llm.clean_transcript, identified_segments
        )
        job_step(job, "transcript_cleaning", f"{len(identified_segments)} segments after cleaning", time.time()-t35, 82)

        # STEP 4 : LLM Summary
        # STEP 4 : LLM Summary
        job["current_step"] = "summary_generation"
        t4 = time.time()
        # Pass enrolled speakers so the LLM uses exact correct spellings
        enrolled_speakers = get_fresh_db().get_all_speakers()
        summary = await loop.run_in_executor(
            None, ComponentRegistry.llm.generate_summary,
            identified_segments, enrolled_speakers
        )
        job_step(job, "summary_generation", f"{len(summary.get('tasks',[]))} tasks extracted", time.time()-t4, 97)

        # Persist result
        # outputs are stored in server temp — final result lives in JOBS dict and localStorage
        outputs_dir = Path(tempfile.gettempdir()) / "xomic_outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_file = outputs_dir / f"{filename}_summary.json"
        with open(output_file, "w") as f:
            json.dump({"summary": summary, "transcript": identified_segments}, f, indent=2, cls=NumpyEncoder)

        total_elapsed = sum(s["elapsed"] for s in job["steps"])
        _print_timing_summary(job["steps"], audio_duration, total_elapsed)

        job["status"]   = "complete"
        job["progress"] = 100
        job["result"]   = {
            "summary":          summary,
            "transcript":       identified_segments,
            "output_file":      str(output_file),
            "audio_duration_s": round(audio_duration, 1),
            "total_pipeline_s": round(total_elapsed, 1),
        }
        log(f"✅ PIPELINE COMPLETE — job={job_id}")

    except Exception as exc:
        job["status"] = "failed"
        job["error"]  = str(exc)
        log(f"❌ PIPELINE FAILED — job={job_id}: {exc}")
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

# ── FIXED /health — now returns models_ready so the welcome page works ──────
@app.get("/health", tags=["System"])
def health():
    ready = all([
        ComponentRegistry.noise_reducer is not None,
        ComponentRegistry.diarization   is not None,
        ComponentRegistry.transcriber   is not None,
        ComponentRegistry.llm           is not None,
        ComponentRegistry.identifier    is not None,
    ])
    return {
        "status":       "ok",
        "version":      "2.0.0",
        "models_ready": ready,   # ← welcome page checks this
    }


@app.get("/config", tags=["System"])
def get_config():
    server = CONFIG.get("server", {})
    host   = server.get("host", "0.0.0.0")
    port   = server.get("port", 8000)
    display_host = "localhost" if host in ("0.0.0.0", "") else host
    return {
        "whisper":     CONFIG.get("whisper", {}),
        "diarization": CONFIG.get("diarization", {}),
        "enrollment":  CONFIG.get("enrollment", {}),
        "server": {
            "url": f"http://{display_host}:{port}",
            "port": port,
        }
    }

@app.get("/paths", tags=["System"])
def get_paths():
    """
    Returns ONLY server-side paths (embeddings + enrollments).
    recordings and outputs are local to each user's machine —
    the Tauri frontend resolves those via the OS path API.
    """
    base = Path(__file__).parent
    def resolve(rel_path):
        p = Path(rel_path)
        if not p.is_absolute(): p = base / p
        return str(p.resolve())
    return {
        "speaker_embeddings": resolve(CONFIG["paths"]["speaker_embeddings"]),
        "enrollments":        resolve(CONFIG["paths"]["enrollments"]),
    }


# ── Enrollment ────────────────────────────────────────────────────────────

@app.get("/speakers", tags=["Enrollment"])
def list_speakers():
    db  = get_fresh_db()
    raw = db.get_all_speakers()
    # Remove embedding array from response (too large for JSON)
    speakers = [{k: v for k, v in s.items() if k != "embedding"} for s in raw]
    return {"speakers": speakers, "count": len(speakers)}


@app.post("/speakers/enroll", tags=["Enrollment"], status_code=status.HTTP_201_CREATED)
async def enroll_speaker(
    name:       str        = Form(...),
    role:       str        = Form(""),
    audio_file: UploadFile = File(...),
):
    enr_dir = Path(CONFIG["paths"]["enrollments"])
    enr_dir.mkdir(parents=True, exist_ok=True)
    save_path = enr_dir / audio_file.filename

    with open(save_path, "wb") as f:
        f.write(await audio_file.read())

    success, message = await asyncio.get_event_loop().run_in_executor(
        None, ComponentRegistry.enrollment.enroll_speaker,
        name, str(save_path), role or None,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {"success": True, "message": message}


@app.delete("/speakers/{speaker_id}", tags=["Enrollment"])
def delete_speaker(speaker_id: str):
    db      = get_fresh_db()
    deleted = db.delete_speaker(speaker_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")
    return {"success": True, "deleted_id": speaker_id}


# ── Analysis ──────────────────────────────────────────────────────────────

@app.post("/analyze", tags=["Analysis"], status_code=status.HTTP_202_ACCEPTED)
async def analyze_meeting(
    background_tasks: BackgroundTasks,
    audio_file: UploadFile = File(...),
):
    # recordings are transient — use a server-side temp folder, not config
    recordings_dir = Path(tempfile.gettempdir()) / "xomic_recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    save_path = recordings_dir / audio_file.filename

    with open(save_path, "wb") as f:
        f.write(await audio_file.read())

    if audio_file.filename.lower().endswith(".mp4"):
        save_path = Path(convert_mp4_to_wav(str(save_path)))

    job_id = str(uuid.uuid4())
    new_job(job_id, audio_file.filename)

    background_tasks.add_task(run_pipeline, job_id, save_path, audio_file.filename)

    return {
        "job_id":   job_id,
        "status":   "accepted",
        "filename": audio_file.filename,
    }


# ── Jobs ──────────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}", tags=["Jobs"])
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if job["status"] == "complete":
        return job
    return {
        "id":           job["id"],
        "name":         job["name"],
        "status":       job["status"],
        "progress":     job["progress"],
        "current_step": job.get("current_step", ""),
        "steps":        job["steps"],
        "error":        job["error"],
    }


@app.get("/jobs/{job_id}/stream", tags=["Jobs"])
async def stream_job(job_id: str):
    """SSE endpoint — frontend listens here for real-time progress updates."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    async def event_generator():
        last_step_count = 0
        while True:
            job   = JOBS.get(job_id, {})
            steps = job.get("steps", [])

            if len(steps) > last_step_count:
                for step in steps[last_step_count:]:
                    data = _json({"type": "progress", **step, "progress": job["progress"]})
                    yield f"data: {data}\n\n"
                last_step_count = len(steps)

            if job.get("status") == "complete":
                yield f"data: {_json({'type': 'complete', 'progress': 100, 'job_id': job_id})}\n\n"
                break
            elif job.get("status") == "failed":
                yield f"data: {_json({'type': 'error', 'error': job.get('error', 'Unknown error')})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/jobs", tags=["Jobs"])
def list_jobs():
    summaries = [
        {"id": j["id"], "name": j["name"], "status": j["status"],
         "progress": j["progress"], "created_at": j["created_at"]}
        for j in JOBS.values()
    ]
    return {"jobs": sorted(summaries, key=lambda j: j["created_at"], reverse=True)}


@app.get("/jobs/{job_id}/download", tags=["Jobs"])
def download_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "complete":
        raise HTTPException(status_code=400, detail="Job not yet complete")
    output_file = job["result"].get("output_file")
    if not output_file or not Path(output_file).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(
        output_file,
        media_type="application/json",
        filename=f"meeting_summary_{job['name']}.json",
    )

# ── Entry point — reads host/port from config.yaml ────────────────────────
if __name__ == "__main__":
    import uvicorn
    _server = CONFIG.get("server", {})
    _host   = _server.get("host", "0.0.0.0")
    _port   = _server.get("port", 8000)
    logger.info(f"\n🚀  Starting on http://localhost:{_port}\n")
    uvicorn.run("api:app", host=_host, port=_port, reload=True)