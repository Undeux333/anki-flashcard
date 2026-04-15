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

PROP_PHRASE        = "Phrase"
PROP_STATUS        = "Status"
PROP_GENERATED_AT  = "Generated At"
PROP_RELEASE_URL   = "Release URL"

STATUS_READY   = "Ready"
STATUS_DONE    = "Done"
STATUS_ERROR   = "Error"
STATUS_TIMEOUT = "Timeout"
STATUS_PENDING = "Pending"

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
audio { display: none; }
.replay-button { display: none; }
.main-btn-wrap { display: flex; justify-content: center; margin-top: 20px; }
.main-btn { display: flex; align-items: center; justify-content: center; gap: 10px; padding: 14px 32px; background: #2d3748; border-radius: 12px; cursor: pointer; border: none; }
.main-btn-text { font-size: 15px; font-weight: bold; color: #fff; }
.conv { display: flex; flex-direction: column; gap: 8px; margin-bottom: 8px; }
.conv-row { display: flex; align-items: center; gap: 0px; }
.conv-speaker { font-size: 12px; font-weight: bold; color: #718096; flex-shrink: 0; }
.conv-bar { height: 38px; border-radius: 8px; background: #edf2f7; border: 1px solid #e2e8f0; flex: 1; }
.conv-predict { min-height: 38px; border-radius: 8px; background: #edf2f7; border: 1.5px solid #f6c026; flex: 1; display: flex; align-items: center; padding: 6px 12px; gap: 7px; font-size: 12px; color: #4a5568; }
.line-block { margin-bottom: 10px; }
.line-row { display: flex; align-items: center; gap: 0px; }
.back-speaker { font-size: 12px; font-weight: bold; color: #718096; flex-shrink: 0; }
.bubble { flex: 1; background: #fff; border: 0.5px solid #e2e8f0; border-radius: 10px; padding: 10px 14px; font-size: 15px; line-height: 1.5; color: #2d3748; }
.bubble.predict { border: 1.5px solid #f6c026; }
.bubble b { color: #000; font-weight: bold; border-bottom: 2px solid #cbd5e0; }
.action-row { display: flex; align-items: center; justify-content: flex-end; gap: 6px; margin-top: 5px; }
.explain-btn { font-size: 11px; color: #2b6cb0; cursor: pointer; font-weight: bold; padding: 4px 9px; border-radius: 10px; white-space: nowrap; border: 1px solid #bee3f8; background: #ebf4ff; }
.script-icon-btn { font-size: 14px; cursor: pointer; padding: 3px 7px; border-radius: 8px; border: 1px solid #bee3f8; background: #ebf4ff; line-height: 1.4; }
.script-icon-btn.on { background: #2b6cb0; border-color: #2b6cb0; }
.slow-btn { font-size: 11px; color: #4a5568; cursor: pointer; font-weight: bold; padding: 4px 9px; background: #f7fafc; border-radius: 10px; white-space: nowrap; border: 1px solid #e2e8f0; }
.play-line-btn { font-size: 11px; color: #4a5568; cursor: pointer; font-weight: bold; padding: 4px 9px; background: #f7fafc; border-radius: 10px; white-space: nowrap; border: 1px solid #e2e8f0; }
.explain-box { margin-top: 5px; background: #ebf4ff; border-left: 3px solid #4299e1; border-radius: 0 8px 8px 0; padding: 9px 12px; font-size: 13px; color: #2c5282; font-style: italic; line-height: 1.55; display: none; }
.rl { font-size: 10px; font-weight: bold; padding: 2px 8px; border-radius: 10px; }
.rn { background: #e8f4fd; color: #1d6fa4; }
"""

# ═══════════════════════════════════════════════
#   ロジック
# ═══════════════════════════════════════════════
def get_speech_lines(phrase):
    """A: / B: / A?: / B?: を解析し hidden フラグを付与する"""
    raw_parts = re.split(r'(A\?:|B\?:|A:|B:)', phrase)
    lines = []
    current_speaker = "A"
    current_hidden = False
    for part in raw_parts:
        p = part.strip()
        if not p:
            continue
        if p == "A:":
            current_speaker = "A"; current_hidden = False
        elif p == "B:":
            current_speaker = "B"; current_hidden = False
        elif p == "A?:":
            current_speaker = "A"; current_hidden = True
        elif p == "B?:":
            current_speaker = "B"; current_hidden = True
        else:
            lines.append({"speaker": current_speaker, "text": p, "hidden": current_hidden})
    return lines

def generate_content(client, speech_lines: list) -> dict:
    label_count = len(speech_lines)
    input_text = "\n".join([f"{l['speaker']}: {l['text']}" for l in speech_lines])

    hidden_indices = [i for i, l in enumerate(speech_lines) if l['hidden']]
    hidden_info = f"Hidden phrase indices (0-based): {hidden_indices}" if hidden_indices else "No hidden phrases."

    prompt = f"""Explain the nuance of EACH phrase marked with A: or B: in the following dialogue. 
Explain how this phrase is used in everyday conversation by native speakers by briefly describing the situation or feeling it is used for. Focus on the speaker's feeling and intention. Keep it short, simple, and natural in one sentence. Use plain, everyday English. Avoid abstract or textbook-like language, and avoid extra details or unnecessary assumptions. Write as if explaining to an English learner in a casual conversation.
CRITICAL RULE: 
The input has exactly {label_count} labeled phrases. 
You MUST provide exactly {label_count} explanations in the "meanings" array. 
One explanation per phrase, in the same order.

For hidden phrases, also provide a short hint in "hints" telling the learner what kind of response to give.
The hint must be a natural English instruction, max 10 words. For non-hidden phrases, use null.
{hidden_info}

Input:
{input_text}

Return ONLY valid JSON: 
{{
  "meanings": ["explanation for phrase 1", "explanation for phrase 2", ...],
  "hints": [null, "hint for hidden phrase", null, ...]
}}"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            )
            data = json.loads(response.text.strip())
            while len(data["meanings"]) < label_count:
                data["meanings"].append("(Check original text for nuance)")
            data["meanings"] = data["meanings"][:label_count]
            # hints の長さを補正
            hints = data.get("hints", [])
            while len(hints) < label_count:
                hints.append(None)
            data["hints"] = hints[:label_count]
            return data
        except Exception as e:
            if "429" in str(e) or "503" in str(e):
                time.sleep(25)
                continue
            raise e

async def _tts_bytes(text, voice):
    communicate = edge_tts.Communicate(text, voice=voice, rate=TTS_RATE)
    data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            data += chunk["data"]
    return data

async def process_audio(speech_lines: list, meanings: list, uid: str, tmpdir: str):
    s_files, m_files = [], []
    front_audio = AudioSegment.empty()
    back_audio  = AudioSegment.empty()
    last_idx = len(speech_lines) - 1

    for idx, line in enumerate(speech_lines):
        clean_text = re.sub(r'\(|\)', '', line['text'])
        voice = CONV_VOICES.get(line['speaker'], "en-US-BrianNeural")

        s_data = await _tts_bytes(clean_text, voice)
        s_fn = f"s_{uid}_{idx}.mp3"
        (Path(tmpdir) / s_fn).write_bytes(s_data)
        s_files.append(s_fn)

        seg = AudioSegment.from_file(io.BytesIO(s_data), format="mp3")
        trailing = AudioSegment.silent(duration=200) if idx < last_idx else AudioSegment.empty()

        # 表面用: 非表示行は最初・最後を除き無音に置き換え
        if line['hidden']:
            if idx > 0 and idx < last_idx:
                front_audio += AudioSegment.silent(duration=len(seg)) + trailing
        else:
            front_audio += seg + trailing

        # 裏面用: 全行を均一なポーズで結合
        back_audio += seg + trailing

        m_data = await _tts_bytes(meanings[idx], CONV_VOICES["B"])
        m_fn = f"m_{uid}_{idx}.mp3"
        (Path(tmpdir) / m_fn).write_bytes(m_data)
        m_files.append(m_fn)

    f_fn = f"front_{uid}.mp3"
    front_audio.export(str(Path(tmpdir) / f_fn), format="mp3")
    b_fn = f"back_{uid}.mp3"
    back_audio.export(str(Path(tmpdir) / b_fn), format="mp3")
    return f_fn, b_fn, s_files, m_files

def format_script_text(text: str) -> str:
    t = text.replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r'\((.*?)\)', r'<b>\1</b>', t).replace("\n", "<br>")

def build_front(f_fn, speech_lines, hints):
    rows = ""
    for idx, line in enumerate(speech_lines):
        sp = line['speaker']
        if line['hidden']:
            hint_text = hints[idx] if hints and idx < len(hints) and hints[idx] else None
            if hint_text is None:
                raise ValueError(f"hint missing for hidden line {idx}: {line['text']}")
            rows += (
                f'<div class="conv-row">'
                f'<span class="conv-speaker">{sp}:</span>'
                f'<div class="conv-predict"><span style="font-size:14px;">&#127919;</span>{hint_text}</div>'
                f'</div>'
            )
        else:
            rows += (
                f'<div class="conv-row">'
                f'<span class="conv-speaker">{sp}:</span>'
                f'<div class="conv-bar"></div>'
                f'</div>'
            )

    return (
        f'<div class="ep-front">'
        f'[sound:{f_fn}]'
        f'<div class="conv">{rows}</div>'
        f'<div class="main-btn-wrap">'
        f'<div class="main-btn" onclick="document.getElementById(\'fa1\').play()">'
        f'<span style="font-size:18px;">&#128266;</span>'
        f'<span class="main-btn-text">Play conversation</span>'
        f'<audio id="fa1" src="{f_fn}"></audio>'
        f'</div>'
        f'</div>'
        f'</div>'
    )

def build_back(speech_lines, s_files, m_files, meanings, b_fn):
    rows = ""
    for idx, line in enumerate(speech_lines):
        disp = format_script_text(line['text'])
        sp   = line['speaker']
        mt   = meanings[idx]
        bubble_class = "bubble predict" if line['hidden'] else "bubble"
        rows += (
            f'<div class="line-block">'
            f'<div class="line-row">'
            f'<span class="back-speaker">{sp}:</span>'
            f'<div class="{bubble_class}">{disp}</div>'
            f'<audio id="s{idx}" src="{s_files[idx]}"></audio>'
            f'</div>'
            f'<div class="action-row">'
            f'<div class="explain-btn" onclick="document.getElementById(\'m{idx}\').play()">&#128266; Explain</div>'
            f'<div class="script-icon-btn" onclick="epToggle(this,\'ex{idx}\')">&#128196;</div>'
            f'<div class="slow-btn" onclick="epSlow(\'s{idx}\')">&#128034; Slow</div>'
            f'<div class="play-line-btn" onclick="epPlay(\'s{idx}\')">&#128266; Play</div>'
            f'<audio id="m{idx}" src="{m_files[idx]}"></audio>'
            f'</div>'
            f'<div class="explain-box" id="ex{idx}"><i>{mt}</i></div>'
            f'</div>'
        )

    toggle_js = (
        '<script>'
        'function epToggle(btn,id){'
        'var box=document.getElementById(id);'
        'var on=btn.classList.contains("on");'
        'if(on){btn.classList.remove("on");box.style.display="none";}'
        'else{btn.classList.add("on");box.style.display="block";}'
        '}'
        'function epPlay(id){'
        'var a=document.getElementById(id);'
        'a.playbackRate=1.0;a.play();'
        '}'
        'function epSlow(id){'
        'var a=document.getElementById(id);'
        'a.playbackRate=0.5;a.play();'
        '}'
        '</script>'
    )

    return (
        f'<div class="ep-back">'
        f'<div style="margin-bottom:10px"><span class="rl rn">&#9679; Script &amp; Nuance</span></div>'
        f'{rows}'
        f'<div class="main-btn-wrap">'
        f'<div class="main-btn" onclick="document.getElementById(\'ba1\').play()">'
        f'<span style="font-size:18px;">&#128266;</span>'
        f'<span class="main-btn-text">Play all</span>'
        f'<audio id="ba1" src="{b_fn}"></audio>'
        f'</div>'
        f'</div>'
        f'{toggle_js}'
        f'</div>'
    )

# ═══════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════
def main():
    current_dir = Path(__file__).parent.absolute()
    output_path = current_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)

    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    filter_payload = {
        "filter": {
            "or": [
                {"property": PROP_STATUS, "select": {"equals": STATUS_READY}},
                {"property": PROP_STATUS, "select": {"equals": STATUS_TIMEOUT}}
            ]
        }
    }
    if FORCE_REGEN:
        filter_payload = {}

    res = requests.post(
        url,
        headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"},
        json=filter_payload,
        timeout=15
    )
    res.raise_for_status()

    pending = []
    for page in res.json().get("results", []):
        try:
            phrase = page["properties"][PROP_PHRASE]["title"][0]["text"]["content"].strip()
            if phrase:
                pending.append({"phrase": phrase, "page_id": page["id"]})
        except:
            continue

    if not pending:
        print("処理対象のデータ（Ready または Timeout）が見つかりませんでした。")
        return

    model = genanki.Model(
        ANKI_MODEL_ID, "EP_Model_v18",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[{"name": "Card 1", "qfmt": "{{Front}}", "afmt": "{{Back}}"}],
        css=CARD_CSS
    )
    deck = genanki.Deck(ANKI_DECK_ID, "English Phrases (Auto)")
    all_media = []
    done_page_ids = []  # リリースURL記録用

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, item in enumerate(pending, 1):
            phrase, page_id = item["phrase"], item["page_id"]
            print(f"[{i}/{len(pending)}] {phrase[:50]}...")
            uid = hashlib.md5(phrase.encode()).hexdigest()[:10]
            try:
                if i > 1:
                    time.sleep(12)
                speech_lines = get_speech_lines(phrase)
                content = generate_content(client, speech_lines)
                meanings = content["meanings"]
                hints = content.get("hints", [])

                f_fn, b_fn, s_files, m_files = asyncio.run(
                    process_audio(speech_lines, meanings, uid, tmpdir)
                )
                all_media.extend([str(Path(tmpdir) / f) for f in [f_fn, b_fn] + s_files + m_files])

                deck.add_note(genanki.Note(
                    model=model,
                    fields=[
                        build_front(f_fn, speech_lines, hints),
                        build_back(speech_lines, s_files, m_files, meanings, b_fn)
                    ],
                    guid=uid
                ))

                requests.patch(
                    f"https://api.notion.com/v1/pages/{page_id}",
                    headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"},
                    json={"properties": {PROP_STATUS: {"select": {"name": STATUS_DONE}}}},
                    timeout=10
                )
                done_page_ids.append(page_id)
                print("    ✅ 成功")

            except Exception as e:
                err_s = str(e).lower()
                status_to_set = STATUS_TIMEOUT if ("429" in err_s or "503" in err_s or "timeout" in err_s or "deadline" in err_s) else STATUS_ERROR
                print(f"    ❌ 失敗({status_to_set}): {e}")
                requests.patch(
                    f"https://api.notion.com/v1/pages/{page_id}",
                    headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"},
                    json={"properties": {PROP_STATUS: {"select": {"name": status_to_set}}}},
                    timeout=10
                )

        if all_media:
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            final_name = output_path / f"anki_cards_{timestamp}.apkg"
            pkg = genanki.Package(deck)
            pkg.media_files = all_media
            pkg.write_to_file(str(final_name))
            print(f"📦 生成完了: {final_name}")

            # 成功済みページに Generated At を記録
            if done_page_ids:
                generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                for page_id in done_page_ids:
                    try:
                        requests.patch(
                            f"https://api.notion.com/v1/pages/{page_id}",
                            headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"},
                            json={"properties": {PROP_GENERATED_AT: {"date": {"start": generated_at}}}},
                            timeout=10
                        )
                    except Exception as e:
                        print(f"⚠️  Notion Generated At 更新失敗 ({page_id}): {e}")

                # ワークフローが Release URL を記録するためにページIDを保存
                done_pages_path = output_path / "done_pages.json"
                done_pages_path.write_text(json.dumps(done_page_ids))

if __name__ == "__main__":
    main()
