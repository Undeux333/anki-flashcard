#!/usr/bin/env python3
"""
Anki Flashcard Generator v7 — Notion + GitHub Actions
New register system: Neutral / Polite / Casual
"""

import os, json, hashlib, asyncio, tempfile, requests, io, re
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
import edge_tts
import genanki
from pydub import AudioSegment

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
GEMINI_MODEL       = "gemini-3.1-flash-lite-preview"
TTS_RATE           = "+0%"
NOTION_VERSION     = "2022-06-28"
OUTPUT_DIR         = Path("output")

PROP_PHRASE    = "Phrase"
PROP_STATUS    = "Status"
STATUS_PENDING = "Pending"
STATUS_DONE    = "Done"
STATUS_ERROR   = "Error"

ANKI_MODEL_ID = 1607392330
ANKI_DECK_ID  = 2059400110

VOICES = [
    {"id": "brian",   "name": "Brian",   "voice": "en-US-BrianNeural",           "gender": "male",   "desc": "deep"},
    {"id": "ava",     "name": "Ava",     "voice": "en-US-AvaMultilingualNeural", "gender": "female", "desc": "natural"},
    {"id": "steffan", "name": "Steffan", "voice": "en-US-SteffanNeural",         "gender": "male",   "desc": "casual"},
    {"id": "jenny",   "name": "Jenny",   "voice": "en-US-JennyNeural",           "gender": "female", "desc": "clear"},
    {"id": "ana",     "name": "Ana",     "voice": "en-US-AnaNeural",             "gender": "female", "desc": "bright"},
]

CONV_VOICES = {
    "female": "en-US-AvaMultilingualNeural",
    "male":   "en-US-BrianNeural",
}
CONV_VOICES_ALT = {
    "female": "en-US-JennyNeural",
    "male":   "en-US-SteffanNeural",
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
    payload = {
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
            if not title_arr:
                continue
            phrase = title_arr[0]["text"]["content"].strip()
            if phrase:
                results.append({"phrase": phrase, "page_id": page["id"]})
        except (KeyError, IndexError):
            continue
    return results

def update_notion_status(page_id, status):
    url = f"https://api.notion.com/v1/pages/{page_id}"
    payload = {"properties": {PROP_STATUS: {"select": {"name": status}}}}
    requests.patch(url, headers=notion_headers(), json=payload, timeout=10)

# ═══════════════════════════════════════════════
#   Gemini content generation
# ═══════════════════════════════════════════════
def generate_content(client, phrase: str) -> dict:
    prompt = f"""You are a native American English teacher making Anki flashcards for Japanese adult learners.

Target phrase: "{phrase}"

REGISTER SYSTEM — understand this deeply before writing:

● Neutral — The default. How Americans talk to almost everyone.
  Not too casual, not too polite. Natural compression, contractions, normal rhythm.
  Who: coworkers, new people, most everyday situations.
  Key trait: this is the baseline — what most natives say most of the time.

● Polite — A little more considerate. Slightly softer edge.
  NOT longer or more formal — just a bit gentler in tone.
  You still compress. You don't explain more than needed.
  Who: boss, clients, someone you want to be careful with.
  Key trait: short stays short — you soften the angle, not the length.
  Example: "Can you check this?" → Polite → "Could you check this?" or "Mind taking a look?"
  AVOID: over-explaining, long justifications, stiff phrasing.

● Casual — Compressed and fast. Main difference is you DROP words.
  Subject can disappear. Rhythm gets lighter and looser.
  Who: close friends, family — people you're totally relaxed with.
  Key trait: "Can you check this?" → Casual → "Check this out." / "Take a look?"

CRITICAL RULES:
- ALL English must sound like real American speech — never textbook, never written.
- Use contractions, fillers (y'know, I mean, honestly, like), natural rhythm.
- simple_meaning: explain to a native English-speaking child (CEFR A1-A2 vocabulary only). Short, concrete, vivid. Max 2 sentences.
- For time-related phrases: always give SPECIFIC ranges (e.g. "about 3 to 6 hours ago"). Never say "a long time."
- conversations: 3 separate scenes — Neutral first, then Polite, then Casual. Each is a short natural A/B dialogue (3-5 lines). The phrase must appear naturally in context.
- who_to_use: evaluate each register with "best" or "ok". Use "ok" only when the phrase works but needs a small tweak to feel right. Add a short note ONLY for "ok".
- also_say: generate ONLY for registers that are "ok". Skip registers that are "best". If ALL are "best", return an empty array.

Return ONLY valid JSON (no markdown, no backticks):
{{
  "phrase_display": "the phrase with correct capitalization",
  "audio_text": "1 natural Neutral sentence containing the phrase",
  "simple_meaning": "explain to a native English-speaking child in A1-A2 words. Max 2 sentences. Concrete and vivid.",
  "conversations": [
    {{
      "register": "neutral",
      "setting": "brief situation description",
      "speaker_a": {{"gender": "female or male"}},
      "speaker_b": {{"gender": "female or male"}},
      "lines": [
        {{"speaker": "A", "text": "natural line"}},
        {{"speaker": "B", "text": "natural line"}},
        {{"speaker": "A", "text": "natural line using the phrase"}},
        {{"speaker": "B", "text": "natural line"}}
      ]
    }},
    {{
      "register": "polite",
      "setting": "different situation requiring a bit more care",
      "speaker_a": {{"gender": "female or male"}},
      "speaker_b": {{"gender": "female or male"}},
      "lines": [
        {{"speaker": "A", "text": "natural line"}},
        {{"speaker": "B", "text": "natural polite line using the phrase"}},
        {{"speaker": "A", "text": "natural line"}},
        {{"speaker": "B", "text": "natural line"}}
      ]
    }},
    {{
      "register": "casual",
      "setting": "relaxed situation with close friends or family",
      "speaker_a": {{"gender": "female or male"}},
      "speaker_b": {{"gender": "female or male"}},
      "lines": [
        {{"speaker": "A", "text": "casual line"}},
        {{"speaker": "B", "text": "casual compressed line using the phrase"}},
        {{"speaker": "A", "text": "casual line"}},
        {{"speaker": "B", "text": "casual line"}}
      ]
    }}
  ],
  "who_to_use": [
    {{"register": "neutral", "status": "best or ok", "note": "short note only if ok, else empty string"}},
    {{"register": "polite",  "status": "best or ok", "note": "short note only if ok, else empty string"}},
    {{"register": "casual",  "status": "best or ok", "note": "short note only if ok, else empty string"}}
  ],
  "also_say": [
    {{"register": "neutral or polite or casual", "phrase": "alternative spoken phrase", "example": "punchy realistic sentence using the alternative", "note": "1 line: how it feels different from the original"}}
  ],
  "level": "beginner or intermediate or advanced"
}}"""

    import time
    for attempt in range(4):
        try:
            if attempt > 0:
                wait = 30 * attempt
                print(f"    ⏳ Gemini混雑 — {wait}秒待機してリトライ ({attempt}/3)...")
                time.sleep(wait)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.8
                )
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception as e:
            if attempt == 3:
                raise e

# ═══════════════════════════════════════════════
#   Audio generation
# ═══════════════════════════════════════════════
async def _tts_save(text, voice, rate, path):
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(path)

async def _tts_bytes(text, voice, rate):
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(tmp.name)
    data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)
    return data

def generate_all_audio(text: str, uid: str, tmpdir: str) -> list[dict]:
    results = []
    for v in VOICES:
        filename = f"ep_{uid}_{v['id']}.mp3"
        filepath = str(Path(tmpdir) / filename)
        try:
            asyncio.run(_tts_save(text, v["voice"], TTS_RATE, filepath))
            results.append({"voice": v, "filename": filename, "filepath": filepath})
        except Exception as e:
            print(f"    ⚠️  {v['name']} 失敗: {e}")
    return results

def generate_conversation_audio(conv: dict, uid: str, idx: int, tmpdir: str) -> tuple[str, str]:
    gender_a = conv["speaker_a"]["gender"]
    gender_b = conv["speaker_b"]["gender"]
    voice_a = CONV_VOICES.get(gender_a, CONV_VOICES["male"])
    if gender_a == gender_b:
        voice_b = CONV_VOICES_ALT.get(gender_b, CONV_VOICES_ALT["male"])
    else:
        voice_b = CONV_VOICES.get(gender_b, CONV_VOICES["female"])

    silence = AudioSegment.silent(duration=50)
    combined = AudioSegment.silent(duration=100)
    for ln in conv["lines"]:
        voice = voice_a if ln["speaker"] == "A" else voice_b
        try:
            audio_bytes = asyncio.run(_tts_bytes(ln["text"], voice, TTS_RATE))
            segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            combined += segment + silence
        except Exception as e:
            print(f"    ⚠️  会話行生成失敗: {e}")

    filename = f"ep_{uid}_conv{idx}.mp3"
    filepath = str(Path(tmpdir) / filename)
    combined.export(filepath, format="mp3")
    return filename, filepath

# ═══════════════════════════════════════════════
#   HTML builders
# ═══════════════════════════════════════════════
LEVEL_STYLE = {
    "beginner":     "background:#e8f5e9;color:#2e7d32",
    "intermediate": "background:#fff8e1;color:#f57f17",
    "advanced":     "background:#fce4ec;color:#c62828"
}

REG_META = {
    "neutral": {"cls": "rn", "lbl": "● Neutral", "cb": "cn"},
    "polite":  {"cls": "rp", "lbl": "● Polite",  "cb": "cp"},
    "casual":  {"cls": "rc", "lbl": "● Casual",  "cb": "cc"},
}

def highlight_phrase(text: str, phrase: str) -> str:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    return pattern.sub(
        f'<span class="hl">{phrase}</span>',
        text
    )

def build_voice_buttons_front(audio_list):
    buttons = ""
    for item in audio_list:
        v = item["voice"]
        f = item["filename"]
        bg = "#ebf4ff" if v["gender"] == "male" else "#f0fff4"
        fg = "#2b6cb0" if v["gender"] == "male" else "#276749"
        buttons += f'<div class="vbtn"><div class="vp" style="background:{bg};color:{fg};">&#9654;</div><div><div class="vn">{v["name"]}</div><div class="vd">{v["gender"]} &middot; {v["desc"]}</div></div>[sound:{f}]</div>'
    return buttons

def build_voice_buttons_back(audio_list):
    buttons = ""
    for i, item in enumerate(audio_list):
        v = item["voice"]
        f = item["filename"]
        bg = "#ebf4ff" if v["gender"] == "male" else "#f0fff4"
        fg = "#2b6cb0" if v["gender"] == "male" else "#276749"
        aid = f"va_{i}"
        buttons += f'<div class="vbtn" onclick="document.getElementById(\'{aid}\').play()" style="cursor:pointer;"><div class="vp" style="background:{bg};color:{fg};">&#9654;</div><div><div class="vn">{v["name"]}</div><div class="vd">{v["gender"]} &middot; {v["desc"]}</div></div><audio id="{aid}" src="{f}"></audio></div>'
    return buttons

def build_front(audio_list, content):
    lvl = content.get("level", "intermediate")
    style = LEVEL_STYLE.get(lvl, LEVEL_STYLE["intermediate"])
    buttons = build_voice_buttons_front(audio_list)
    return f"""<div class="ep-front">
<span class="ep-badge" style="{style}">{lvl.upper()}</span>
<div class="sec-label">&#9835; Listen &amp; recall</div>
<div class="vgrid">{buttons}</div>
</div>"""

def build_back(audio_list, conv_files, content):
    lvl   = content.get("level", "intermediate")
    style = LEVEL_STYLE.get(lvl, LEVEL_STYLE["intermediate"])
    phrase = content["phrase_display"]

    # Sentence (Neutral)
    audio_text_hl = highlight_phrase(content["audio_text"], phrase)
    brian_filename = audio_list[0]["filename"] if audio_list else ""
    brian_btn = f'<div class="splay" onclick="document.getElementById(\'brian_a\').play()">&#9654;</div><audio id="brian_a" src="{brian_filename}"></audio>' if brian_filename else ""

    # Conversations
    convs_html = ""
    for conv, (conv_filename, _) in zip(content["conversations"], conv_files):
        reg = conv.get("register", "neutral")
        r = REG_META.get(reg, REG_META["neutral"])
        lines_html = ""
        for ln in conv["lines"]:
            sc = "sa" if ln["speaker"] == "A" else "sb2"
            text_hl = highlight_phrase(ln["text"], phrase)
            lines_html += f'<p><span class="{sc}">{ln["speaker"]}:</span> {text_hl}</p>'
        cid = f"conv_{conv_filename}"
        convs_html += f'''<span class="rl {r['cls']}">{r['lbl']}</span>
<div class="cb {r['cb']}">[sound:{conv_filename}]
  <div class="ch"><div class="cs">&#128205; {conv['setting']}</div><div class="cplay">&#9654; Play</div></div>
  <div class="cl">{lines_html}</div>
</div>'''

    # Who to use
    who_html = '<div class="who-tbl">'
    for w in content["who_to_use"]:
        reg = w.get("register", "neutral")
        r = REG_META.get(reg, REG_META["neutral"])
        st = w.get("status", "best")
        if st == "best":
            st_html = '<span class="st sbest">⭐ Best</span>'
        else:
            st_html = '<span class="st sok">🟢 OK</span>'
        note_html = f'<div class="wd">{w["note"]}</div>' if st == "ok" and w.get("note") else ""
        who_html += f'<div class="wr"><div class="wl"><span class="rl {r["cls"]}" style="margin:0">{r["lbl"]}</span></div>{st_html}{note_html}</div>'
    who_html += '</div>'

    # Also say (only if not all best)
    also_html = ""
    also_items = content.get("also_say", [])
    if also_items:
        also_html = '<div class="sec-label">&#128260; What to say instead — and why it\'s different</div>'
        for a in also_items:
            reg = a.get("register", "neutral")
            r = REG_META.get(reg, REG_META["neutral"])
            phrase_hl = highlight_phrase(a["phrase"], a["phrase"])
            ex_hl = highlight_phrase(a.get("example", ""), a["phrase"])
            also_html += f'''<div class="ab">
<span class="rl {r['cls']}">{r['lbl']}</span>
<div class="ap">{phrase_hl}</div>
<div class="ae">{ex_hl}</div>
<div class="an">&#8594; {a['note']}</div>
</div>'''

    # Voice buttons (back — click only)
    buttons_back = build_voice_buttons_back(audio_list)

    return f"""<div class="ep-back">
<span class="ep-badge" style="{style}">{lvl.upper()}</span>
<div class="sentence-wrap">
  <div class="sentence"><div style="margin-bottom:5px"><span class="rl rn">● Neutral</span></div>{audio_text_hl}</div>
  {brian_btn}
</div>
<div class="sec-label">&#128161; What it actually means</div>
<div class="box meaning">{content['simple_meaning']}</div>
<div class="sec-label">&#128172; Real-life conversations</div>
{convs_html}
<div class="sec-label">&#128100; Who can you say this to?</div>
{who_html}
{also_html}
<div class="divider"></div>
<div class="sec-label">&#9835; Listen in all voices</div>
<div class="vgrid">{buttons_back}</div>
</div>"""

CARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
.card {
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
  font-size: 15px; line-height: 1.6; color: #1a1a2e;
  background: #f4f6f9; min-height: 100vh;
}
.ep-front { max-width: 560px; margin: 0 auto; padding: 24px 16px; text-align: center; }
.ep-back  { max-width: 560px; margin: 0 auto; padding: 16px 16px 28px; }
.ep-badge { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; padding: 3px 12px; border-radius: 20px; margin-bottom: 16px; }
.hl { color: #2b6cb0; font-weight: 700; background: #ebf4ff; padding: 0 2px; border-radius: 3px; }
.sec-label { font-size: 10px; font-weight: 700; color: #8a9ab5; letter-spacing: 1px; text-transform: uppercase; margin: 14px 0 8px; }
.vgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
.vbtn { display: flex; align-items: center; gap: 7px; padding: 7px 9px; border: 1px solid #e2e8f0; border-radius: 8px; background: #fff; }
.vp { width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 10px; }
.vn { font-size: 12px; font-weight: 700; color: #2d3748; }
.vd { font-size: 10px; color: #a0aec0; }
.rl { display: inline-flex; align-items: center; font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 20px; margin-bottom: 6px; }
.rn { background: #e8f4fd; color: #1d6fa4; }
.rp { background: #fef9e7; color: #7d4e00; }
.rc { background: #e8f5e9; color: #2e7d32; }
.sentence-wrap { display: flex; align-items: center; gap: 8px; margin: 0 0 14px; }
.sentence { font-size: 14px; line-height: 1.7; color: #2d3748; padding: 10px 14px; background: #fff; border-radius: 8px; border: 1px solid #e2e8f0; flex: 1; }
.splay { width: 32px; height: 32px; border-radius: 50%; background: #ebf4ff; color: #2b6cb0; display: flex; align-items: center; justify-content: center; cursor: pointer; flex-shrink: 0; font-size: 12px; border: 1px solid #bee3f8; }
.box { border-radius: 0 8px 8px 0; padding: 9px 12px; font-size: 13px; line-height: 1.6; }
.meaning { background: #ebf4ff; border-left: 2px solid #4299e1; }
.cb { border-radius: 0 8px 8px 0; margin-bottom: 10px; }
.cn { background: #f0f7ff; border-left: 2px solid #4299e1; }
.cp { background: #fefce8; border-left: 2px solid #d97706; }
.cc { background: #f0fff4; border-left: 2px solid #48bb78; }
.ch { display: flex; align-items: center; justify-content: space-between; padding: 7px 12px 5px; border-bottom: 1px solid rgba(0,0,0,0.06); }
.cs { font-size: 11px; color: #718096; font-style: italic; flex: 1; }
.cplay { font-size: 11px; color: #2b6cb0; font-weight: 700; cursor: pointer; padding-left: 8px; }
.cl { padding: 7px 12px 9px; }
.cl p { font-size: 13px; margin-bottom: 4px; line-height: 1.5; }
.sa { color: #2b6cb0; font-weight: 700; }
.sb2 { color: #c05621; font-weight: 700; }
.who-tbl { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }
.wr { display: flex; align-items: center; gap: 10px; padding: 7px 10px; border-bottom: 1px solid #f0f4f8; }
.wr:last-child { border-bottom: none; }
.wl { flex-shrink: 0; width: 66px; }
.wd { color: #718096; font-size: 11px; flex: 1; }
.st { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 20px; flex-shrink: 0; }
.sbest { background: #d8f3dc; color: #2d6a4f; }
.sok { background: #e8f4fd; color: #1d6fa4; }
.ab { margin-bottom: 9px; }
.ap { font-size: 13px; font-weight: 700; color: #2d3748; margin-bottom: 3px; }
.ae { font-size: 12px; color: #718096; line-height: 1.5; margin-bottom: 2px; font-style: italic; }
.an { font-size: 11px; color: #a0aec0; }
.divider { border: none; border-top: 1px solid #e2e8f0; margin: 14px 0 10px; }
"""

def build_anki_model():
    return genanki.Model(
        ANKI_MODEL_ID,
        "EP_EnglishPhrase_v7",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[{"name": "Card 1", "qfmt": "{{Front}}", "afmt": "{{Back}}"}],
        css=CARD_CSS
    )

def main():
    for var in ["GEMINI_API_KEY", "NOTION_TOKEN", "NOTION_DATABASE_ID"]:
        if not os.environ.get(var):
            print(f"❌ 環境変数 {var} が設定されていません")
            return

    OUTPUT_DIR.mkdir(exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)

    print("📋 Notion から未処理フレーズを取得中...")
    try:
        pending = get_pending_phrases()
    except Exception as e:
        print(f"❌ Notion API エラー: {e}")
        return

    if not pending:
        print("✅ 未処理のフレーズはありません。")
        return

    print(f"   {len(pending)} 件見つかりました\n")

    anki_model = build_anki_model()
    deck = genanki.Deck(ANKI_DECK_ID, "🇺🇸 English Phrases")
    notes = []
    all_media = []
    ok = ng = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, item in enumerate(pending, 1):
            phrase  = item["phrase"]
            page_id = item["page_id"]
            uid     = hashlib.md5(phrase.lower().encode()).hexdigest()[:10]
            print(f"[{i}/{len(pending)}] 📝 {phrase}")

            try:
                print("  ⏳ Gemini でコンテンツ生成中...")
                content = generate_content(client, phrase)

                print("  🎙️  音声を生成中...")
                audio_list = generate_all_audio(content["audio_text"], uid, tmpdir)
                all_media.extend([a["filepath"] for a in audio_list])

                print("  🎭  会話音声を3つ生成中...")
                conv_files = []
                for idx, conv in enumerate(content["conversations"], 1):
                    filename, filepath = generate_conversation_audio(conv, uid, idx, tmpdir)
                    conv_files.append((filename, filepath))
                    all_media.append(filepath)

                front = build_front(audio_list, content)
                back  = build_back(audio_list, conv_files, content)

                note = genanki.Note(
                    model=anki_model,
                    fields=[front, back],
                    guid=uid,
                    tags=["auto", "english-phrase"]
                )
                notes.append(note)
                deck.add_note(note)
                update_notion_status(page_id, STATUS_DONE)
                print(f"  ✅ 完了")
                ok += 1

            except Exception as e:
                print(f"  ❌ エラー: {e}")
                update_notion_status(page_id, STATUS_ERROR)
                ng += 1

        if notes:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"english_phrases_{timestamp}.apkg"
            package = genanki.Package(deck)
            package.media_files = all_media
            package.write_to_file(str(output_path))
            print(f"\n📦 .apkg 生成完了: {output_path}")

    print(f"\n{'━'*46}")
    print(f"🎉 完了: {ok} 枚作成 / {ng} 件エラー")

if __name__ == "__main__":
    main()
