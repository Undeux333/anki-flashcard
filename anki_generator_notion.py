#!/usr/bin/env python3
import os, json, hashlib, asyncio, tempfile, requests, io, re, time
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
import edge_tts
import genanki
from pydub import AudioSegment

# ═══════════════════════════════════════════════
#   設定・環境変数
# ═══════════════════════════════════════════════
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
GEMINI_MODEL       = "gemini-3.1-flash-lite-preview"
TTS_RATE           = "+0%"
NOTION_VERSION     = "2022-06-28"

PROP_PHRASE    = "Phrase"
PROP_STATUS    = "Status"
STATUS_DONE    = "Done"
STATUS_ERROR   = "Error"

ANKI_MODEL_ID = 1607392333
ANKI_DECK_ID  = 2059400110
FORCE_REGEN = os.environ.get("FORCE_REGEN", "false").lower() == "true"

CONV_VOICES = {
    "A": "en-US-BrianNeural",
    "B": "en-US-AvaMultilingualNeural",
}

CARD_CSS = """
.card { font-family: sans-serif; background: #f4f6f9; text-align: left; }
.ep-front, .ep-back { max-width: 550px; margin: auto; padding: 20px; }
.sentence-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 20px; }
.splay-wrap { display: flex; flex-direction: column; gap: 5px; flex-shrink: 0; }
.splay { width: 55px; height: 32px; border-radius: 16px; background: #fff; display: flex; align-items: center; justify-content: center; cursor: pointer; border: 1px solid #e2e8f0; font-size: 10px; font-weight: bold; color: #4a5568; }
.sentence-content { flex: 1; min-width: 0; }
.sentence { font-size: 17px; padding: 12px; background: #fff; border-radius: 8px; color: #2d3748; line-height: 1.5; border: 1px solid #e2e8f0; }
.sentence b { color: #000; font-weight: bold; border-bottom: 2px solid #cbd5e0; }
.meaning-box { display: flex; align-items: flex-end; justify-content: space-between; margin-top: 8px; background: #ebf4ff; padding: 10px; border-radius: 8px; border-left: 4px solid #4299e1; }
.mini-meaning { font-size: 14px; color: #2c5282; line-height: 1.4; flex: 1; }
.mplay { font-size: 11px; color: #3182ce; cursor: pointer; font-weight: bold; padding: 4px 8px; }


