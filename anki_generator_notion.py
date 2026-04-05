#!/usr/bin/env python3
"""
Anki Flashcard Generator — Notion + GitHub Actions 版
Notion DB の「Pending」フレーズを取得 → Gemini + edge-tts → .apkg 生成
"""

import os, json, hashlib, asyncio, tempfile, requests
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
import edge_tts
import genanki

GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
GEMINI_MODEL       = "gemini-3.1-flash-lite"
TTS_VOICE          = "en-US-AndrewNeural"
TTS_RATE           = "-12%"
NOTION_VERSION     = "2022-06-28"
OUTPUT_DIR         = Path("output")

PROP_PHRASE    = "Phrase"
PROP_STATUS    = "Status"
STATUS_PENDING = "Pending"
STATUS_DONE    = "Done"
STATUS_ERROR   = "Error"

ANKI_MODEL_ID = 1607392319
ANKI_DECK_ID  = 2059400110

def notion_headers():
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json"
    }

def get_pending_phrases():
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    payload = {"filter": {"property": PROP_STATUS, "select": {"equals": STATUS_PENDING}}}
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
    prompt = f"""You are a professional American English teacher creating a detailed flashcard.

Target phrase or expression: "{phrase}"

Return ONLY a valid JSON object (no markdown, no backticks) with this exact structure:
{{
  "phrase_display": "the phrase exactly as it should appear",
  "audio_text": "natural phrase or sentence for TTS",
  "simple_meaning": "Clear 1-2 sentence explanation in simple English (B1 level).",
  "conversation": {{
    "setting": "One sentence describing the real-life situation",
    "lines": [
      {{"speaker": "A", "text": "...", "note": ""}},
      {{"speaker": "B", "text": "...", "note": "brief note or empty string"}},
      {{"speaker": "A", "text": "...", "note": ""}},
      {{"speaker": "B", "text": "...", "note": ""}}
    ]
  }},
  "situations": ["Specific situation 1", "Specific situation 2", "Specific situation 3"],
  "paraphrases": ["Alternative phrase 1 with brief note", "Alternative phrase 2 with brief note"],
  "tips": "1-2 critical tips about register, common mistakes, or American culture.",
  "level": "beginner or intermediate or advanced"
}}"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.7
        )
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

async def _tts(text, voice, rate, path):
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate)
    await communicate.save(path)

def generate_audio(text: str, uid: str, tmpdir: str) -> tuple[str, str]:
    filename = f"ep_{uid}.mp3"
    filepath = str(Path(tmpdir) / filename)
    asyncio.run(_tts(text, TTS_VOICE, TTS_RATE, filepath))
    return filepath, filename

LEVEL_STYLE = {
    "beginner":     "background:#e8f5e9;color:#2e7d32",
    "intermediate": "background:#fff8e1;color:#f57f17",
    "advanced":     "background:#fce4ec;color:#c62828"
}

def build_front(audio_file, content):
    lvl = content.get("level", "intermediate")
    style = LEVEL_STYLE.get(lvl, LEVEL_STYLE["intermediate"])
    return f"""<div class="ep-front">
<span class="ep-badge" style="{style}">{lvl.upper()}</span>
<div class="ep-audio-area">
  <div class="ep-play">&#9654;</div>
  <div class="ep-hint">Native American English &middot; Male</div>
</div>
[sound:{audio_file}]
</div>"""

def build_back(audio_file, content):
    conv  = content["conversation"]
    lines = ""
    for ln in conv["lines"]:
        cls  = "ep-sa" if ln["speaker"] == "A" else "ep-sb"
        note = f' <span class="ep-note">({ln["note"]})</span>' if ln.get("note") else ""
        lines += f'<p><span class="{cls}">{ln["speaker"]}:</span> {ln["text"]}{note}</p>'
    sits = "".join(f'<div class="ep-item">{s}</div>' for s in content["situations"])
    para = "".join(f'<div class="ep-item">{p}</div>' for p in content["paraphrases"])
    return f"""<div class="ep-back">
<div class="ep-phrase">&#8220;{content['phrase_display']}&#8221;</div>
[sound:{audio_file}]
<div class="ep-label">&#128161; Meaning</div>
<div class="ep-box ep-meaning">{content['simple_meaning']}</div>
<div class="ep-label">&#128172; Conversation</div>
<div class="ep-box ep-conv"><div class="ep-setting">{conv['setting']}</div>{lines}</div>
<div class="ep-label">&#127919; When to use</div>
<div class="ep-box ep-sit">{sits}</div>
<div class="ep-label">&#128260; You can also say</div>
<div class="ep-box ep-para">{para}</div>
<div class="ep-label">&#9889; Tips</div>
<div class="ep-box ep-tips">{content['tips']}</div>
</div>"""

CARD_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
.card {
  font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif;
  font-size: 16px; line-height: 1.65; color: #1a1a2e;
  background: #f4f6f9; min-height: 100vh; padding: 20px 16px;
}
.ep-front, .ep-back { max-width: 560px; margin: 0 auto; }
.ep-badge { display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; padding: 3px 12px; border-radius: 20px; margin-bottom: 20px; }
.ep-audio-area { display: flex; flex-direction: column; align-items: center; padding: 28px 0 20px; }
.ep-play { width: 76px; height: 76px; border-radius: 50%; background: #ffffff; border: 1px solid #dce1ea; display: flex; align-items: center; justify-content: center; font-size: 28px; color: #4a5568; }
.ep-hint { font-size: 12px; color: #8a9ab5; margin-top: 10px; }
.ep-phrase { font-size: 24px; font-weight: 700; color: #2b6cb0; text-align: center; margin-bottom: 16px; }
.ep-label { font-size: 11px; font-weight: 700; color: #8a9ab5; letter-spacing: 1px; text-transform: uppercase; margin: 14px 0 6px; }
.ep-box { border-radius: 8px; padding: 11px 14px; font-size: 14px; }
.ep-meaning { background: #ebf4ff; border-left: 3px solid #4299e1; }
.ep-conv    { background: #f0fff4; border-left: 3px solid #48bb78; }
.ep-sit     { background: #fffbeb; border-left: 3px solid #ecc94b; }
.ep-para    { background: #f5f3ff; border-left: 3px solid #9f7aea; }
.ep-tips    { background: #fff5f5; border-left: 3px solid #fc8181; }
.ep-setting { font-size: 12px; color: #718096; font-style: italic; margin-bottom: 8px; }
.ep-sa { color: #2b6cb0; font-weight: 700; }
.ep-sb { color: #c05621; font-weight: 700; }
.ep-note { color: #a0aec0; font-size: 12px; }
.ep-item { padding: 2px 0 2px 14px; position: relative; }
.ep-item::before { content: "·"; position: absolute; left: 3px; color: #a0aec0; }
p { margin-bottom: 5px; }
"""

def build_anki_model():
    return genanki.Model(
        ANKI_MODEL_ID,
        "EP_EnglishPhrase_v2",
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
    audio_files = []
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

                print("  🎙️  edge-tts で音声生成中...")
                audio_path, audio_filename = generate_audio(content["audio_text"], uid, tmpdir)
                audio_files.append(audio_path)

                front = build_front(audio_filename, content)
                back  = build_back(audio_filename, content)

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
            package.media_files = audio_files
            package.write_to_file(str(output_path))
            print(f"\n📦 .apkg 生成完了: {output_path}")

    print(f"\n{'━'*46}")
    print(f"🎉 完了: {ok} 枚作成 / {ng} 件エラー")

if __name__ == "__main__":
    main()
