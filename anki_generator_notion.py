#!/usr/bin/env python3
"""
Anki Flashcard Generator v6 — Notion + GitHub Actions
変更: 表面は音声のみ・裏面フレーズ削除・会話形式3つ・個別音声ボタン
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
TTS_RATE           = "0%"
NOTION_VERSION     = "2022-06-28"
OUTPUT_DIR         = Path("output")

PROP_PHRASE    = "Phrase"
PROP_STATUS    = "Status"
STATUS_PENDING = "Pending"
STATUS_DONE    = "Done"
STATUS_ERROR   = "Error"

ANKI_MODEL_ID = 1607392322
ANKI_DECK_ID  = 2059400110

VOICES = [
    {"id": "steffan",     "name": "Steffan",     "voice": "en-US-SteffanNeural",         "gender": "male",   "desc": "casual"},
    {"id": "andrew",      "name": "Andrew",      "voice": "en-US-AndrewNeural",          "gender": "male",   "desc": "warm"},
    {"id": "christopher", "name": "Christopher", "voice": "en-US-ChristopherNeural",     "gender": "male",   "desc": "deep"},
    {"id": "eric",        "name": "Eric",        "voice": "en-US-EricNeural",            "gender": "male",   "desc": "clear"},
    {"id": "ava",         "name": "Ava",         "voice": "en-US-AvaMultilingualNeural", "gender": "female", "desc": "natural"},
    {"id": "aria",        "name": "Aria",        "voice": "en-US-AriaNeural",            "gender": "female", "desc": "expressive"},
    {"id": "jenny",       "name": "Jenny",       "voice": "en-US-JennyNeural",           "gender": "female", "desc": "clear"},
    {"id": "ana",         "name": "Ana",         "voice": "en-US-AnaNeural",             "gender": "female", "desc": "bright"},
]

CONV_VOICES = {
    "female": "en-US-JennyNeural",
    "male":   "en-US-AndrewNeural",
}

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

def generate_content(client, phrase: str) -> dict:
    prompt = f"""You are a native American English teacher making Anki flashcards for Japanese adult learners.

Target phrase: "{phrase}"

CRITICAL RULES:
- ALL English must sound 100% like real native American speech. No textbook English ever.
- Use contractions, fillers (y'know, I mean, honestly, like), natural flow.
- Slang is OK. Offensive language is NOT OK.
- simple_meaning: explain like talking to a 10-year-old American kid. Super simple, max 2 sentences. Be concrete and vivid.
- For time-related phrases, always give SPECIFIC time ranges (e.g. "about 3 to 6 hours ago", "roughly 2 weeks"). Never say vague things like "a long time".
- conversations: 3 completely different real-life situations. Each must be a SHORT natural A/B conversation (3-5 lines) that naturally includes the phrase. 100% colloquial American English.
- also_say: for each relationship type, give the alternative phrase AND explain in 1 line HOW it feels different.
- related: for each phrase, explain exactly WHEN or HOW to use it in 1 concrete line.

Return ONLY valid JSON (no markdown, no backticks):
{{
  "phrase_display": "the phrase with correct capitalization",
  "audio_text": "1 natural sentence that contains the phrase in context",
  "simple_meaning": "explain like to a 10-year-old American kid, super concrete, max 2 sentences.",
  "conversations": [
    {{
      "setting": "brief situation description",
      "speaker_a": {{"name": "first name", "gender": "female or male"}},
      "speaker_b": {{"name": "first name", "gender": "female or male"}},
      "lines": [
        {{"speaker": "A", "text": "natural colloquial line"}},
        {{"speaker": "B", "text": "natural colloquial line"}},
        {{"speaker": "A", "text": "natural colloquial line using the phrase"}},
        {{"speaker": "B", "text": "natural colloquial line"}}
      ]
    }},
    {{
      "setting": "completely different situation",
      "speaker_a": {{"name": "first name", "gender": "female or male"}},
      "speaker_b": {{"name": "first name", "gender": "female or male"}},
      "lines": [
        {{"speaker": "A", "text": "natural colloquial line"}},
        {{"speaker": "B", "text": "natural colloquial line using the phrase"}},
        {{"speaker": "A", "text": "natural colloquial line"}},
        {{"speaker": "B", "text": "natural colloquial line"}}
      ]
    }},
    {{
      "setting": "completely different situation",
      "speaker_a": {{"name": "first name", "gender": "female or male"}},
      "speaker_b": {{"name": "first name", "gender": "female or male"}},
      "lines": [
        {{"speaker": "A", "text": "natural colloquial line"}},
        {{"speaker": "B", "text": "natural colloquial line"}},
        {{"speaker": "A", "text": "natural colloquial line"}},
        {{"speaker": "B", "text": "natural colloquial line using the phrase"}}
      ]
    }}
  ],
  "who_to_use": [
    {{"who": "Close friends", "status": "best or ok or avoid", "note": "short exception note if ok, else empty string"}},
    {{"who": "Friends", "status": "best or ok or avoid", "note": ""}},
    {{"who": "Family", "status": "best or ok or avoid", "note": ""}},
    {{"who": "Coworkers", "status": "best or ok or avoid", "note": ""}},
    {{"who": "Boss / Formal", "status": "best or ok or avoid", "note": ""}}
  ],
  "also_say": [
    {{"who": "Close friends / Family", "phrase": "alternative phrase", "note": "exactly how it feels different in 1 line"}},
    {{"who": "Friends / Coworkers", "phrase": "alternative phrase", "note": "exactly how it feels different in 1 line"}},
    {{"who": "Boss / Formal", "phrase": "alternative phrase", "note": "exactly how it feels different in 1 line"}}
  ],
  "related": [
    {{"phrase": "related phrase", "note": "exactly when/how to use it in 1 concrete line", "priority": "must"}},
    {{"phrase": "related phrase", "note": "exactly when/how to use it in 1 concrete line", "priority": "must"}},
    {{"phrase": "related phrase", "note": "exactly when/how to use it in 1 concrete line", "priority": "soon"}},
    {{"phrase": "related phrase", "note": "exactly when/how to use it in 1 concrete line", "priority": "soon"}},
    {{"phrase": "related phrase", "note": "exactly when/how to use it in 1 concrete line", "priority": "later"}}
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
    voice_a = CONV_VOICES.get(conv["speaker_a"]["gender"], CONV_VOICES["female"])
    voice_b = CONV_VOICES.get(conv["speaker_b"]["gender"], CONV_VOICES["male"])
    silence = AudioSegment.silent(duration=600)
    combined = AudioSegment.silent(duration=200)
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

def highlight_phrase(text: str, phrase: str) -> str:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    return pattern.sub(
        f'<span class="ep-highlight">{phrase}</span>',
        text
    )

def build_voice_buttons(audio_list):
    buttons = ""
    for item in audio_list:
        v = item["voice"]
        f = item["filename"]
        bg = "#ebf4ff" if v["gender"] == "male" else "#f0fff4"
        fg = "#2b6cb0" if v["gender"] == "male" else "#276749"
        buttons += f"""<div class="ep-vbtn"><div class="ep-vplay" style="background:{bg};color:{fg};">&#9654;</div><div class="ep-vinfo"><div class="ep-vname">{v['name']}</div><div class="ep-vdesc">{v['gender']} &middot; {v['desc']}</div></div>[sound:{f}]</div>"""
    return buttons

def build_front(audio_list, content):
    lvl = content.get("level", "intermediate")
    style = LEVEL_STYLE.get(lvl, LEVEL_STYLE["intermediate"])
    buttons = build_voice_buttons(audio_list)
    return f"""<div class="ep-front">
<span class="ep-badge" style="{style}">{lvl.upper()}</span>
<div class="ep-label">&#127925; Listen &amp; recall</div>
<div class="ep-vgrid">{buttons}</div>
</div>"""

def build_back(audio_list, conv_files, content):
    lvl   = content.get("level", "intermediate")
    style = LEVEL_STYLE.get(lvl, LEVEL_STYLE["intermediate"])

    # 会話3つ（個別音声ボタン付き）
    convs_html = ""
    for i, (conv, (conv_filename, _)) in enumerate(zip(content["conversations"], conv_files), 1):
        name_a = conv["speaker_a"]["name"]
        name_b = conv["speaker_b"]["name"]
        lines_html = ""
        for ln in conv["lines"]:
            name = name_a if ln["speaker"] == "A" else name_b
            cls  = "ep-sa" if ln["speaker"] == "A" else "ep-sb"
            text_hl = highlight_phrase(ln["text"], content["phrase_display"])
            lines_html += f'<p><span class="{cls}">{name}:</span> {text_hl}</p>'

        convs_html += f"""<div class="ep-conv-block">
  <div class="ep-conv-header">
    <div class="ep-conv-setting">&#128205; {conv['setting']}</div>
    <div class="ep-conv-play-btn">[sound:{conv_filename}] &#9654;</div>
  </div>
  <div class="ep-lines">{lines_html}</div>
</div>"""

    # Who to use
    who_rows = ""
    for w in content["who_to_use"]:
        icon, label, fg, bg = STATUS_MAP.get(w["status"], STATUS_MAP["ok"])
        note_html = f' <span class="ep-tbl-note">· {w["note"]}</span>' if w.get("note") else ""
        who_rows += f'<tr><td class="ep-tbl-who">{w["who"]}</td><td><span class="ep-status" style="color:{fg};background:{bg};">{icon} {label}</span>{note_html}</td></tr>'

    # Also say
    also_html = ""
    for a in content["also_say"]:
        also_html += f'<div class="ep-also-row"><div class="ep-also-who">{a["who"]}</div><div class="ep-also-phrase">{a["phrase"]}</div><div class="ep-also-note">&#8594; {a["note"]}</div></div>'

    # Related
    related_html = ""
    for r in content["related"]:
        fg, bg = PRIO_STYLE[r["priority"]][1], PRIO_STYLE[r["priority"]][2]
        label = PRIO_STYLE[r["priority"]][0]
        related_html += f'<div class="ep-rel-row"><span class="ep-prio" style="color:{fg};background:{bg};">{label}</span><span class="ep-rel-phrase">{r["phrase"]}</span><span class="ep-rel-note">&#8594; {r["note"]}</span></div>'

    # 音声ボタン（一番下）
    buttons = build_voice_buttons(audio_list)

    return f"""<div class="ep-back">
<div class="ep-inner">
<span class="ep-badge" style="{style}">{lvl.upper()}</span>

<div class="ep-label">&#128161; What it actually means</div>
<div class="ep-box ep-meaning">{content['simple_meaning']}</div>

<div class="ep-label">&#128172; Real-life conversations</div>
{convs_html}

<div class="ep-label">&#128100; Who can you say this to?</div>
<table class="ep-tbl"><tbody>{who_rows}</tbody></table>

<div class="ep-label">&#128260; What to say instead — and why it's different</div>
<div class="ep-also">{also_html}</div>

<div class="ep-label">&#128218; Learn these next — and exactly when to use them</div>
<div class="ep-related">{related_html}</div>

<div class="ep-label" style="margin-top:20px;border-top:1px solid #e2e8f0;padding-top:14px;">&#127925; Listen to the phrase in all voices</div>
<div class="ep-vgrid">{buttons}</div>
</div></div>"""

LEVEL_STYLE = {
    "beginner":     "background:#e8f5e9;color:#2e7d32",
    "intermediate": "background:#fff8e1;color:#f57f17",
    "advanced":     "background:#fce4ec;color:#c62828"
}

STATUS_MAP = {
    "best":  ("⭐", "Best",  "#2d6a4f", "#d8f3dc"),
    "ok":    ("🟢", "OK",    "#1d6fa4", "#e8f4fd"),
    "avoid": ("🔴", "Avoid", "#9b2226", "#fce4e4"),
}

PRIO_STYLE = {
    "must":  ("must",  "#9b2226", "#fce4e4"),
    "soon":  ("soon",  "#7d4e00", "#fff3cd"),
    "later": ("later", "#5f5e5a", "#f1efe8"),
}

CARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
.card {
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
  font-size: 15px; line-height: 1.6; color: #1a1a2e;
  background: #f4f6f9; min-height: 100vh;
}
.ep-front { max-width: 560px; margin: 0 auto; padding: 24px 16px; }
.ep-back { max-width: 560px; margin: 0 auto; }
.ep-inner { padding: 16px 16px 24px; }
.ep-badge { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; padding: 3px 12px; border-radius: 20px; margin-bottom: 16px; }
.ep-highlight { color: #2b6cb0; font-weight: 700; background: #ebf4ff; padding: 0 2px; border-radius: 3px; }
.ep-label { font-size: 10px; font-weight: 700; color: #8a9ab5; letter-spacing: 1px; text-transform: uppercase; margin: 16px 0 8px; }
.ep-vgrid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
.ep-vbtn { display: flex; align-items: center; gap: 7px; padding: 7px 9px; border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff; }
.ep-vplay { width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 10px; }
.ep-vname { font-size: 12px; font-weight: 700; color: #2d3748; }
.ep-vdesc { font-size: 10px; color: #a0aec0; }
.ep-box { border-radius: 0 8px 8px 0; padding: 9px 12px; font-size: 13px; }
.ep-meaning { background: #ebf4ff; border-left: 3px solid #4299e1; }
.ep-conv-block { background: #f0fff4; border-left: 3px solid #48bb78; border-radius: 0 8px 8px 0; margin-bottom: 10px; overflow: hidden; }
.ep-conv-header { display: flex; align-items: center; justify-content: space-between; padding: 8px 12px 6px; border-bottom: 1px solid #c6f6d5; }
.ep-conv-setting { font-size: 11px; color: #276749; font-style: italic; flex: 1; }
.ep-conv-play-btn { font-size: 12px; color: #2b6cb0; font-weight: 700; white-space: nowrap; padding-left: 8px; }
.ep-lines { padding: 8px 12px 10px; }
.ep-lines p { font-size: 13px; margin-bottom: 5px; line-height: 1.55; }
.ep-sa { color: #2b6cb0; font-weight: 700; }
.ep-sb { color: #c05621; font-weight: 700; }
.ep-tbl { width: 100%; border-collapse: collapse; font-size: 12px; background: #fff; border-radius: 8px; overflow: hidden; border: 1px solid #e2e8f0; }
.ep-tbl-who { padding: 7px 10px; color: #4a5568; border-bottom: 1px solid #f0f4f8; width: 45%; }
.ep-tbl tr:last-child td { border-bottom: none; }
.ep-tbl td:last-child { padding: 7px 10px; border-bottom: 1px solid #f0f4f8; }
.ep-status { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 20px; }
.ep-tbl-note { font-size: 11px; color: #718096; }
.ep-also { display: flex; flex-direction: column; gap: 6px; }
.ep-also-row { background: #fff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 10px; }
.ep-also-who { font-size: 10px; color: #8a9ab5; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 2px; }
.ep-also-phrase { font-size: 13px; font-weight: 700; color: #2d3748; margin-bottom: 3px; }
.ep-also-note { font-size: 11px; color: #718096; }
.ep-related { display: flex; flex-direction: column; gap: 5px; }
.ep-rel-row { display: flex; align-items: baseline; gap: 7px; font-size: 12px; flex-wrap: wrap; padding: 4px 0; }
.ep-prio { font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 20px; white-space: nowrap; }
.ep-rel-phrase { font-weight: 700; color: #2d3748; }
.ep-rel-note { color: #718096; font-size: 11px; }
"""

def build_anki_model():
    return genanki.Model(
        ANKI_MODEL_ID,
        "EP_EnglishPhrase_v6",
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

                print("  🎙️  8音声を生成中...")
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
