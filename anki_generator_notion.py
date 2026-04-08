#!/usr/bin/env python3
"""
Anki Flashcard Generator v9 — Notion + Dialogue Support
Features: 
- Fixed phrase usage (No modifications by AI)
- A: B: Dialogue audio synthesis (Male/Female mix)
- Bold text for () brackets
- Native-like simple explanations
"""

import os, json, hashlib, asyncio, tempfile, requests, io, re
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
GEMINI_MODEL       = "gemini-2.0-flash" # 最新モデルを推奨
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

# 音声設定
VOICES = [
    {"id": "brian", "name": "Brian", "voice": "en-US-BrianNeural", "gender": "male", "desc": "deep"},
]
CONV_VOICES = {
    "A": "en-US-BrianNeural", # Aさんは男性
    "B": "en-US-AvaMultilingualNeural", # Bさんは女性
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
#   Gemini content generation
# ═══════════════════════════════════════════════
def generate_content(client, phrase: str) -> dict:
    prompt = f"""You are a helpful native American teacher.
Explain the following English phrase to a 10-year-old native child.

Target: "{phrase}"

INSTRUCTIONS:
- Simple Vocabulary: Use CEFR A1-A2 level words only.
- Contextual: Explain *when* and *why* we say this, not just the literal meaning.
- Nuance: Briefly mention the tone (e.g., "it's a bit funny" or "it's polite").
- Style: Keep it to 2 short sentences.
- Dialogue: If it contains "A:" and "B:", explain the situation between the two people.

Return ONLY valid JSON (no markdown):
{{
  "simple_meaning": "The explanation here",
  "level": "beginner or intermediate or advanced"
}}"""

    for attempt in range(3):
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
            content["phrase_display"] = phrase # Notionのフレーズをそのまま使用
            return content
        except Exception as e:
            if attempt == 2: raise e
            asyncio.run(asyncio.sleep(5))

# ═══════════════════════════════════════════════
#   Audio generation (Dialogue Support)
# ═══════════════════════════════════════════════
async def _tts_bytes(text, voice, rate):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(tmp.name)
    data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)
    return data

async def generate_dialogue_audio(phrase: str, uid: str, tmpdir: str):
    filename = f"ep_{uid}_main.mp3"
    filepath = Path(tmpdir) / filename
    
    # A: B: で分割 (カッコ内の文字も対象)
    parts = re.split(r'(A:|B:)', phrase)
    
    if len(parts) > 1:
        # 会話モード
        combined = AudioSegment.empty()
        current_speaker = "A"
        for part in parts:
            clean_part = part.strip()
            if clean_part == "A:":
                current_speaker = "A"
                continue
            if clean_part == "B:":
                current_speaker = "B"
                continue
            if not clean_part:
                continue
            
            # カッコを削除して読み上げ
            speech_text = re.sub(r'\(|\)', '', clean_part)
            voice = CONV_VOICES[current_speaker]
            audio_data = await _tts_bytes(speech_text, voice, TTS_RATE)
            segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            combined += segment + AudioSegment.silent(duration=400)
        
        combined.export(str(filepath), format="mp3")
    else:
        # 通常モード (カッコを削除して読み上げ)
        speech_text = re.sub(r'\(|\)', '', phrase)
        communicate = edge_tts.Communicate(speech_text, voice=VOICES[0]["voice"], rate=TTS_RATE)
        await communicate.save(str(filepath))
        
    return [{"voice": VOICES[0], "filename": filename, "filepath": str(filepath)}]

# ═══════════════════════════════════════════════
#   HTML builders & Formatter
# ═══════════════════════════════════════════════
def format_script_text(text: str) -> str:
    """() を太字に変換し、改行を保持する"""
    # エスケープ処理
    t = text.replace("<", "&lt;").replace(">", "&gt;")
    # カッコ内を太字
    t = re.sub(r'\((.*?)\)', r'<b>\1</b>', t)
    # 改行対応
    t = t.replace("\n", "<br>")
    return t

def build_front(audio_list):
    filename = audio_list[0]["filename"]
    return f"""<div class="ep-front">
[sound:{filename}]
<div class="sec-label">&#9835; Listen &amp; recall</div>
<div class="vgrid">
  <div class="vbtn" onclick="document.getElementById('fa_1').play()">
    <div class="vp" style="background:#ebf4ff;color:#2b6cb0;">&#9654;</div>
    <div><div class="vn">Play Audio</div><div class="vd">American English</div></div>
    <audio id="fa_1" src="{filename}"></audio>
  </div>
</div>
</div>"""

def build_back(audio_list, content):
    display_phrase = format_script_text(content["phrase_display"])
    filename = audio_list[0]["filename"]
    
    return f"""<div class="ep-back">
<div class="sentence-wrap">
  <div class="sentence">
    <div style="margin-bottom:5px"><span class="rl rn">● Script</span></div>
    {display_phrase}
  </div>
  <div class="splay" onclick="document.getElementById('ba_1').play()">&#9654;</div>
  <audio id="ba_1" src="{filename}"></audio>
</div>
<div class="divider"></div>
<div class="sec-label">&#128214; What it means</div>
<div class="box meaning">{content['simple_meaning']}</div>
</div>"""

CARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
.card { font-family: -apple-system, sans-serif; font-size: 15px; background: #f4f6f9; color: #1a1a2e; }
.ep-front, .ep-back { max-width: 560px; margin: 0 auto; padding: 24px 16px; }
.sec-label { font-size: 10px; font-weight: 700; color: #8a9ab5; text-transform: uppercase; margin-bottom: 8px; }
.divider { border-top: 1px solid #e2e8f0; margin: 16px 0; }
.vgrid { display: flex; justify-content: center; }
.vbtn { display: flex; align-items: center; gap: 10px; padding: 10px 20px; border: 1px solid #e2e8f0; border-radius: 12px; background: #fff; cursor: pointer; }
.vp { width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; }
.vn { font-size: 13px; font-weight: 700; }
.vd { font-size: 11px; color: #a0aec0; }
.sentence { font-size: 15px; color: #2d3748; padding: 12px; background: #fff; border-radius: 8px; border: 1px solid #e2e8f0; flex: 1; line-height: 1.6; }
.sentence b { color: #1a1a2e; font-weight: 800; }
.sentence-wrap { display: flex; align-items: center; gap: 10px; }
.splay { width: 36px; height: 36px; border-radius: 50%; background: #ebf4ff; color: #2b6cb0; display: flex; align-items: center; justify-content: center; cursor: pointer; border: 1px solid #bee3f8; }
.rl { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
.rn { background: #e8f4fd; color: #1d6fa4; }
.box { padding: 12px; border-radius: 8px; font-size: 14px; line-height: 1.5; }
.meaning { background: #ebf4ff; border-left: 4px solid #4299e1; }
"""

# ═══════════════════════════════════════════════
#   Main Process
# ═══════════════════════════════════════════════
def main():
    for var in ["GEMINI_API_KEY", "NOTION_TOKEN", "NOTION_DATABASE_ID"]:
        if not os.environ.get(var):
            print(f"❌ {var} not set"); return

    OUTPUT_DIR.mkdir(exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)

    pending = get_pending_phrases()
    if not pending:
        print("✅ No pending phrases."); return

    anki_model = genanki.Model(ANKI_MODEL_ID, "EP_Model_v9",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[{"name": "Card 1", "qfmt": "{{Front}}", "afmt": "{{Back}}"}],
        css=CARD_CSS)
    
    deck = genanki.Deck(ANKI_DECK_ID, "🇺🇸 English Phrases (Auto)")
    all_media = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, item in enumerate(pending, 1):
            phrase, page_id = item["phrase"], item["page_id"]
            uid = hashlib.md5(phrase.encode()).hexdigest()[:10]
            print(f"[{i}/{len(pending)}] Processing: {phrase}")

            try:
                content = generate_content(client, phrase)
                audio_list = asyncio.run(generate_dialogue_audio(phrase, uid, tmpdir))
                all_media.extend([a["filepath"] for a in audio_list])

                note = genanki.Note(model=anki_model, 
                                    fields=[build_front(audio_list), build_back(audio_list, content)],
                                    guid=uid)
                deck.add_note(note)
                update_notion_status(page_id, STATUS_DONE)
                print("  ✅ Success")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                update_notion_status(page_id, STATUS_ERROR)

        if all_media:
            pkg = genanki.Package(deck)
            pkg.media_files = all_media
            out = OUTPUT_DIR / f"deck_{datetime.now().strftime('%m%d_%H%M')}.apkg"
            pkg.write_to_file(str(out))
            print(f"\n📦 Generated: {out}")

if __name__ == "__main__":
    main()
