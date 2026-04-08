#!/usr/bin/env python3
"""
Anki Flashcard Generator v12 — Final Robust Version
- Gemini 3.1 Flash Lite (429 Retry)
- Dual Audio (Sentence & Explanation)
- Nuance-focused Prompt
- Fixed variable scoping (CARD_CSS)
"""

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

# ═══════════════════════════════════════════════
#   CSS 定義 (エラー防止のため上部で定義)
# ═══════════════════════════════════════════════
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
.mplay { font-size: 11px; color: #3182ce; cursor: pointer; font-weight: bold; padding: 4px 8px; background: #fff; border-radius: 12px; margin-left: 10px; flex-shrink: 0; white-space: nowrap; border: 1px solid #bee3f8; }
.rl { font-size: 10px; font-weight: bold; padding: 2px 8px; border-radius: 10px; }
.rn { background: #e8f4fd; color: #1d6fa4; }
.vbtn { display: flex; align-items: center; background: #fff; padding: 10px 15px; border-radius: 10px; cursor: pointer; border: 1px solid #e2e8f0; width: fit-content; font-size: 14px; }
.vp { width: 24px; height: 24px; border-radius: 50%; background: #ebf4ff; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-size: 10px; }
"""

# ═══════════════════════════════════════════════
#   各関数 (Notion, Gemini, Audio)
# ═══════════════════════════════════════════════
def notion_headers():
    return {"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}

def get_pending_phrases():
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    payload = {} if FORCE_REGEN else {"filter": {"or": [{"property": PROP_STATUS, "select": {"is_empty": True}}, {"property": PROP_STATUS, "select": {"equals": STATUS_ERROR}}]}}
    res = requests.post(url, headers=notion_headers(), json=payload, timeout=15)
    res.raise_for_status()
    results = []
    for page in res.json().get("results", []):
        try:
            phrase = page["properties"][PROP_PHRASE]["title"][0]["text"]["content"].strip()
            if phrase: results.append({"phrase": phrase, "page_id": page["id"]})
        except: continue
    return results

def update_notion_status(page_id, status):
    requests.patch(f"https://api.notion.com/v1/pages/{page_id}", headers=notion_headers(), json={"properties": {PROP_STATUS: {"select": {"name": status}}}}, timeout=10)

def generate_content(client, phrase: str) -> dict:
    prompt = f"""Explain the nuance of this phrase as it is actually used by native speakers in everyday conversation. 
Focus on the speaker’s intention and feeling rather than just the literal meaning. 
Keep the explanation simple, natural, and conversational. 
Avoid over-explaining, exaggeration, or adding unnecessary assumptions.

Target
