#!/usr/bin/env python3
"""
Anki Flashcard Generator v10 — Notion + Dialogue + Robust Error Handling
Features:
- Gemini 3.1 Flash Lite (with exponential backoff for 429 errors)
- Dialogue support (A: Brian / B: Ava)
- Bold text for () brackets
- Enhanced explanation prompt
- Fixed phrase usage (No modifications)
"""

import os, json, hashlib, asyncio, tempfile, requests, io, re, time
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
import edge_tts
import genanki
from pydub import AudioSegment

# 環境変数
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
GEMINI_MODEL       = "gemini-3.1-flash-lite-preview"
TTS_RATE           = "+0%"
NOTION_VERSION     = "2022-06-28"
OUTPUT_DIR         = Path("output")

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
#   Notion API
# ═══════════════════════════════════════════════
def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }

def get_pending_phrases():
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    payload = {} if FORCE_REGEN else {
        "filter": {
            "or": [
                {"property": PROP_STATUS, "select": {"is_empty": True}},
                {"property": PROP_STATUS, "select": {"equals": STATUS_ERROR}}
            ]
        }
    }
    res = requests.post(url, headers=notion_headers(), json=payload, timeout=15)
    res.raise_for_status()
    results = []
    for page in res.json().get("results", []):
        try:
            title_arr = page["properties"][PROP_PHRASE]["title"]
            if not title_arr: continue
            phrase = title_arr[0]["text"]["content"].strip()
            if phrase:
                results.append({"phrase": phrase, "page_id": page["id"]})
        except (KeyError, IndexError): continue
    return results

def update_notion_status(page_id, status):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    payload = {"properties": {PROP_STATUS: {"select": {"name": status}}}}
    requests.patch(url, headers=notion_headers(), json=payload, timeout=10)

# ═══════════════════════════════════════════════
#   Gemini content generation (Rate Limit Handling)
# ═══════════════════════════════════════════════
def generate_content(client, phrase: str) -> dict:
    prompt = f"""You are a helpful native American teacher.
Explain the following English phrase to a 10-year-old native child.
Target: "{phrase}"
INSTRUCTIONS:
- Simple Vocabulary: Use CEFR A1-A2 level words only.
- Contextual: Explain *when* and *why* we say this.
- Nuance: Briefly mention the tone.
- Style: Keep it to 2 short sentences.
- Dialogue: If it contains "A:" and "B:", explain the situation.
Return ONLY valid JSON:
{{
  "simple_meaning": "The explanation here",
  "level": "beginner or intermediate or advanced"
}}"""

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4
                )
            )
            content = json.loads(response.text.strip())
            content["phrase_display"] = phrase
            return content

        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg:
                # 429エラー時は待機時間を指数関数的に増やす、または指示に従う
                wait_match = re.search(r"retry in (\d+\.?\d*)s", err_msg)
                wait_time = float(wait_match.group(1)) if wait_match else (15 * (attempt + 1))
                print(f"  ⚠️ レート制限(429)発生。{wait_time:.1f}秒待機して再試行します ({attempt+1}/{max_retries})...")
                time.sleep(wait_time + 2)
                continue
            raise e
    raise Exception("Gemini APIのリトライ上限に達しました。")

# ═══════════════════════════════════════════════
#   Audio & HTML Processing
# ═══════════════════════════════════════════════
async def _tts_bytes(text, voice, rate):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    try:
        communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
        await communicate.save(tmp.name)
        return Path(tmp.name).read_bytes()
    finally:
        Path(tmp.name).unlink(missing_ok=True)

async def generate_dialogue_audio(phrase: str, uid: str, tmpdir: str):
    filename = f"ep_{uid}_main.mp3"
    filepath = Path(tmpdir) / filename
    parts = re.split(r'(A:|B:)', phrase)
    
    if len(parts) > 1:
        combined = AudioSegment.empty()
        current_speaker = "A"
        for part in parts:
            clean = part.strip()
            if clean == "A:": current_speaker = "A"; continue
            if clean == "B:": current_speaker = "B"; continue
            if not clean: continue
            
            # 音声生成時はカッコを外す
            speech_text = re.sub(r'\(|\)', '', clean)
            voice = CONV_VOICES[current_speaker]
            audio_data = await _tts_bytes(speech_text, voice, TTS_RATE)
            segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            combined += segment + AudioSegment.silent(duration=450)
        combined.export(str(filepath), format="mp3")
    else:
        speech_text = re.sub(r'\(|\)', '', phrase)
        communicate = edge_tts.Communicate(speech_text, voice="en-US-BrianNeural", rate=TTS_RATE)
        await communicate.save(str(filepath))
        
    return [{"filename": filename, "filepath": str(filepath)}]

def format_script_text(text: str) -> str:
    t = text.replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r'\((.*?)\)', r'<b>\1</b>', t)
    return t.replace("\n", "<br>")

def build_front(audio_list):
    fn = audio_list[0]["filename"]
    return f"""<div class="ep-front">[sound:{fn}]<div class="sec-label">Listen</div>
<div class="vbtn" onclick="document.getElementById('fa1').play()"><div class="vp">▶</div><div>Play</div><audio id="fa1" src="{fn}"></audio></div></div>"""

def build_back(audio_list, content):
    p = format_script_text(content["phrase_display"])
    fn = audio_list[0]["filename"]
    return f"""<div class="ep-back">
<div class="sentence-wrap"><div class="sentence">{p}</div><div class="splay" onclick="document.getElementById('ba1').play()">▶</div><audio id="ba1" src="{fn}"></audio></div>
<div class="divider"></div><div class="sec-label">Meaning</div><div class="box meaning">{content['simple_meaning']}</div></div>"""

CARD_CSS = """
.card { font-family: sans-serif; background: #f4f6f9; text-align: left; }
.ep-front, .ep-back { max-width: 500px; margin: auto; padding: 20px; }
.sentence { font-size: 18px; padding: 15px; background: #fff; border-radius: 8px; flex: 1; color: #2d3748; }
.sentence b { color: #000; font-weight: bold; border-bottom: 2px solid #cbd5e0; }
.sentence-wrap { display: flex; align-items: center; gap: 10px; }
.divider { height: 1px; background: #e2e8f0; margin: 15px 0; }
.sec-label { font-size: 11px; color: #a0aec0; margin-bottom: 5px; text-transform: uppercase; }
.box { padding: 12px; border-radius: 8px; font-size: 15px; }
.meaning { background: #ebf4ff; border-left: 4px solid #4299e1; }
.splay { width: 40px; height: 40px; border-radius: 50%; background: #fff; display: flex; align-items: center; justify-content: center; cursor: pointer; border: 1px solid #e2e8f0; }
"""

# ═══════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════
def main():
    client = genai.Client(api_key=GEMINI_API_KEY)
    pending = get_pending_phrases()
    if not pending: return

    model = genanki.Model(ANKI_MODEL_ID, "EP_Model_v10",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[{"name": "Card 1", "qfmt": "{{Front}}", "afmt": "{{Back}}"}], css=CARD_CSS)
    deck = genanki.Deck(ANKI_DECK_ID, "English Phrases (Auto)")
    all_media = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, item in enumerate(pending, 1):
            phrase = item["phrase"]
            print(f"[{i}/{len(pending)}] {phrase[:40]}...")
            uid = hashlib.md5(phrase.encode()).hexdigest()[:10]

            try:
                # レート制限を考慮したインターバル
                if i > 1: time.sleep(10) 
                
                content = generate_content(client, phrase)
                audio_list = asyncio.run(generate_dialogue_audio(phrase, uid, tmpdir))
                all_media.extend([a["filepath"] for a in audio_list])

                deck.add_note(genanki.Note(model=model, fields=[build_front(audio_list), build_back(audio_list, content)], guid=uid))
                update_notion_status(item["page_id"], STATUS_DONE)
                print("  ✅ 成功")

            except Exception as e:
                print(f"  ❌ 失敗: {e}")
                update_notion_status(item["page_id"], STATUS_ERROR)

        if all_media:
            pkg = genanki.Package(deck)
            pkg.media_files = all_media
            out = OUTPUT_DIR / f"deck_{datetime.now().strftime('%m%d%H%M')}.apkg"
            pkg.write_to_file(str(out))
            print(f"📦 生成完了: {out}")

if __name__ == "__main__":
    main()
