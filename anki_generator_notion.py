#!/usr/bin/env python3
"""
Anki Flashcard Generator v8 — Notion + GitHub Actions
Fixes: front autoplay, back click-only, consistent highlight, also_say conversations
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
GEMINI_MODEL          = "gemini-3.1-flash-lite-preview"
TTS_RATE           = "+0%"
NOTION_VERSION     = "2022-06-28"
OUTPUT_DIR         = Path("output")

PROP_PHRASE    = "Phrase"
PROP_STATUS    = "Status"
STATUS_PENDING = "Pending"
STATUS_DONE    = "Done"
STATUS_ERROR   = "Error"

ANKI_MODEL_ID = 1607392333
FORCE_REGEN = os.environ.get("FORCE_REGEN", "false").lower() == "true"
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
    if FORCE_REGEN:
        # No filter — fetch ALL entries regardless of status
        payload = {}
        print("  ⚠️  FORCE_REGEN=true: 全エントリを再生成します (Done含む)")
    else:
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

WHO_TO_USE EVALUATION — do this BEFORE writing anything else:
  STEP 1 — identify the phrase's CORE CONTEXT: What exact moment/situation triggers this phrase?
            Is it a reaction? A discovery? A request? An explanation? Write this out mentally first.
  STEP 2 — rate each register honestly:
    "best"  = sounds completely natural, zero awkwardness, no adjustment needed whatsoever.
    "ok"    = works but slightly off in tone — a small word change makes it noticeably better.
    "avoid" = sounds unnatural, inappropriate, or jarring in this register.
  STEP 3 — reality check: Rating ALL THREE as "best" is almost never correct.
            Most phrases have at least one "ok" or "avoid" register.
            If you rated all three "best", re-examine — you likely missed a nuance.
            Example of correct evaluation for "That's what I was looking for.":
              Neutral → best (natural everyday reaction)
              Polite  → ok (slightly casual for a client/boss setting; better: "That's exactly what I had in mind.")
              Casual  → best (natural with friends)

- audio_text: 1 natural sentence using the ORIGINAL phrase, style matching the FIRST "best" register.

highlight_forms: list EVERY exact form that appears in audio_text. Use straight apostrophes only.

Return ONLY valid JSON (no markdown, no backticks):
{{
  "phrase_display": "the phrase with correct capitalization",
  "audio_text": "1 natural sentence with the ORIGINAL phrase — style matches the first best register",
  "highlight_forms": ["exact forms of audio_text phrase — straight apostrophes"],
  "simple_meaning": "A1-A2 vocabulary. Max 2 sentences. Explain to a native English-speaking child.",
  "highlight_forms": ["exact forms of audio_text phrase — straight apostrophes"],
  "simple_meaning": "A1-A2 vocabulary. Max 2 sentences. Explain to a native English-speaking child.",
  "who_to_use": [
    {{"register": "neutral", "status": "best or ok or avoid", "note": "short note if ok or avoid, else empty string"}},
    {{"register": "polite",  "status": "best or ok or avoid", "note": "short note if ok or avoid, else empty string"}},
    {{"register": "casual",  "status": "best or ok or avoid", "note": "short note if ok or avoid, else empty string"}}
  ],
  "also_say": [
    {{
      "register": "neutral or polite or casual",
      "phrase": "the alternative phrase for this register",
      "highlight_forms": ["exact forms of the ALTERNATIVE phrase as it appears in this register's conversation"],
      "example": "one punchy realistic sentence using the alternative",
      "note": "1 line: how it feels different from the original — A1-A2 vocabulary"
    }}
  ],
  "level": "beginner or intermediate or advanced"
}}"""

    import time
    last_err = None
    for attempt in range(3):
        try:
            if attempt > 0:
                wait = attempt * 15
                print(f"    ⏳ 503混雑 — {wait}秒待機後にリトライ ({attempt}/2)...")
                time.sleep(wait)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.4
                )
            )
            text = response.text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            content = json.loads(text.strip())
            content = _clean_gemini_text(content)
            return _enforce_also_say(content)
        except Exception as e:
            last_err = e
    raise last_err

def _clean_gemini_text(content: dict) -> dict:
    """
    Strip extra whitespace from Gemini-generated text fields.
    temperature=0.8 can occasionally produce extra spaces mid-word.
    """
    import re as _re

    def _clean(s: str) -> str:
        if not isinstance(s, str):
            return s
        # Collapse multiple consecutive spaces into one
        s = _re.sub(r' {2,}', ' ', s)
        return s.strip()

    content["audio_text"]    = _clean(content.get("audio_text", ""))
    content["simple_meaning"] = _clean(content.get("simple_meaning", ""))
    for a in content.get("also_say", []):
        a["phrase"]   = _clean(a.get("phrase", ""))
        a["example"]  = _clean(a.get("example", ""))
        a["note"]     = _clean(a.get("note", ""))
    return content

def _enforce_also_say(content: dict) -> dict:
    """
    Hard-enforces also_say consistency:
      - only keep items whose register is 'ok' or 'avoid' in who_to_use
      - deduplicate by register (keep first occurrence)
      - 'best' registers must never appear in also_say
    """
    alt_registers = {
        w["register"]
        for w in content.get("who_to_use", [])
        if w.get("status") in ("ok", "avoid")
    }
    seen = set()
    filtered = []
    for a in content.get("also_say", []):
        reg = a.get("register", "")
        if reg in alt_registers and reg not in seen:
            filtered.append(a)
            seen.add(reg)
    content["also_say"] = filtered
    return content

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

# ═══════════════════════════════════════════════
#   Highlight helpers
# ═══════════════════════════════════════════════
def _get_phrase_forms(phrase: str) -> list[str]:
    """
    Return the phrase and a punctuation-stripped variant so matching works
    regardless of whether Gemini appended a period/comma to the phrase.
    """
    forms = [phrase]
    stripped = phrase.rstrip('.,!?;:')
    if stripped and stripped != phrase:
        forms.append(stripped)
    return forms

def normalize_text(text: str) -> str:
    """Normalize curly quotes and dashes to ASCII equivalents for matching."""
    return (text
        .replace('\u2019', "'").replace('\u2018', "'")
        .replace('\u201c', '"').replace('\u201d', '"')
        .replace('\u2013', '-').replace('\u2014', '-'))

def highlight_any_form(text: str, forms: list[str]) -> str:
    """
    Single-pass highlight: builds one alternation regex from all forms and applies
    it ONCE to the original text. This prevents any subsequent pass from matching
    inside already-inserted <span> tags and corrupting the HTML.
    Word boundaries are added at word-character edges to prevent mid-word matches.
    """
    if not forms or not text:
        return text
    text_n = normalize_text(text)

    # Deduplicate and discard single-character forms
    seen: set[str] = set()
    cleaned: list[str] = []
    for f in forms:
        fn = normalize_text(f).strip()
        if len(fn) < 3:
            continue
        key = fn.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(fn)

    if not cleaned:
        return text_n

    # Longest first so the alternation regex prefers the longest match
    cleaned.sort(key=len, reverse=True)

    parts: list[str] = []
    for fn in cleaned:
        try:
            escaped = re.escape(fn)
            # Word boundary only at edges that are word characters (letters/digits)
            prefix = r'\b' if fn[0].isalnum() else ""
            suffix = r'\b' if fn[-1].isalnum() else ""
            parts.append(prefix + escaped + suffix)
        except (re.error, IndexError):
            continue

    if not parts:
        return text_n

    try:
        pattern = re.compile("|".join(parts), re.IGNORECASE)
        return pattern.sub(lambda m: f'<span class="hl">{m.group()}</span>', text_n)
    except re.error:
        return text_n

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

def build_voice_buttons_front(audio_list):
    """
    Front: all buttons use onclick + <audio> for manual replay.
    The ONE [sound:brian] tag in build_front() handles the single autoplay.
    """
    buttons = ""
    for item in audio_list:
        v = item["voice"]
        f = item["filename"]
        bg = "#ebf4ff" if v["gender"] == "male" else "#f0fff4"
        fg = "#2b6cb0" if v["gender"] == "male" else "#276749"
        aid = f"fa_{v['id']}"
        buttons += (
            f'<div class="vbtn" onclick="document.getElementById(\'{aid}\').play()" style="cursor:pointer;">'
            f'<div class="vp" style="background:{bg};color:{fg};">&#9654;</div>'
            f'<div><div class="vn">{v["name"]}</div><div class="vd">{v["gender"]} &middot; {v["desc"]}</div></div>'
            f'<audio id="{aid}" src="{f}"></audio>'
            f'</div>'
        )
    return buttons

def build_voice_buttons_back(audio_list):
    """Back: all buttons onclick only, no autoplay."""
    buttons = ""
    for i, item in enumerate(audio_list):
        v = item["voice"]
        f = item["filename"]
        bg = "#ebf4ff" if v["gender"] == "male" else "#f0fff4"
        fg = "#2b6cb0" if v["gender"] == "male" else "#276749"
        aid = f"va_{i}"
        buttons += (
            f'<div class="vbtn" onclick="document.getElementById(\'{aid}\').play()" style="cursor:pointer;">'
            f'<div class="vp" style="background:{bg};color:{fg};">&#9654;</div>'
            f'<div><div class="vn">{v["name"]}</div><div class="vd">{v["gender"]} &middot; {v["desc"]}</div></div>'
            f'<audio id="{aid}" src="{f}"></audio>'
            f'</div>'
        )
    return buttons

def build_front(audio_list, content):
    buttons = build_voice_buttons_front(audio_list)
    autoplay = f'[sound:{audio_list[0]["filename"]}]' if audio_list else ""
    return f"""<div class="ep-front">
{autoplay}
<div class="sec-label">&#9835; Listen &amp; recall</div>
<div class="vgrid">{buttons}</div>
</div>"""


def get_best_reg(content: dict) -> str:
    """Return the first 'best' register, falling back to 'neutral'."""
    who_status = {w["register"]: w.get("status", "best") for w in content.get("who_to_use", [])}
    return next((r for r in ["neutral", "polite", "casual"] if who_status.get(r) == "best"), "neutral")



def build_back(audio_list, content, best_reg: str):
    main_forms = _get_phrase_forms(content["phrase_display"])
    best_r = REG_META.get(best_reg, REG_META["neutral"])

    audio_text_hl = highlight_any_form(content["audio_text"], main_forms)
    brian_f = audio_list[0]["filename"] if audio_list else ""
    brian_btn = (
        f'<div class="splay" onclick="document.getElementById(\'brian_a\').play()">&#9654;</div>'
        f'<audio id="brian_a" src="{brian_f}"></audio>'
    ) if brian_f else ""

    D = '<div class="divider"></div>'
    buttons_back = build_voice_buttons_back(audio_list)

    return f"""<div class="ep-back">
<div class="sentence-wrap">
  <div class="sentence"><div style="margin-bottom:5px"><span class="rl {best_r['cls']}">{best_r['lbl']}</span></div>{audio_text_hl}</div>
  {brian_btn}
</div>
{D}<div class="sec-label">&#128214; What it means</div>
<div class="box meaning">{content['simple_meaning']}</div>
{D}<div class="sec-label">&#9835; Listen in all voices</div>
<div class="vgrid">{buttons_back}</div>
</div>"""

def build_anki_model():
    return genanki.Model(
        ANKI_MODEL_ID,
        "EP_EnglishPhrase_v9",
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

                best_reg = get_best_reg(content)
                front = build_front(audio_list, content)
                back  = build_back(audio_list, content, best_reg)

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
