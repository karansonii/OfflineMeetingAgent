# AI Meeting Agent

AI-powered meeting assistant that performs transcription, speaker diarization, speaker identification, and generates structured summaries using LLM.

---

## Features
- Speech-to-text transcription (Whisper - server)
- Speaker diarization & identification (Pyannote - server)
- AI-generated meeting summaries & action items (LLM)
- Queue-based processing for handling multiple users

---

## Project Setup

Create required folders before running:

```bash
mkdir data
mkdir data/recordings
mkdir data/enrollments
mkdir data/outputs



## Installation
pip install -r requirements.txt

##Run Project (FastAPI)
pip install -r requirements.txt


## Recommend Model for STT
Faster-Whisper-Larger-V3
